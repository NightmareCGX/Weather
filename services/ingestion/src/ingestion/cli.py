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
import concurrent.futures
import json
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

from ingestion.core.base import (
    CycleStoreMismatchError,
    LeadTimeMismatchError,
    PredecessorState,
)
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
    global_members = tuple(_as_list(getattr(args, "member", None)))
    if not models or not dates or not hours or not leads:
        raise ValueError(
            "At least one --model, --cycle-date, --cycle-hour, and "
            "--lead-time-hours is required."
        )
    # Aligned-zipped expansion: the number of (model, date, hour) triples must
    # match unless a single value is broadcast across the others. A single
    # model/date/hour broadcasts; multiple values must align 1:1.
    triples = _align_triples(models, dates, hours)

    def _run_members(model: str) -> tuple[int, ...]:
        """Resolve the member identities for one model in the batch.

        ``--member`` is meaningful only for ensemble models (GEFS) and is
        ignored for deterministic models (GFS). A global ``--member 1 2 3`` in
        a mixed ``--model gfs gefs`` batch must therefore attach members to the
        GEFS run only; the GFS run keeps ``()`` so its store is pre-allocated
        WITHOUT a ``member`` axis (a member-shaped GFS store would reject every
        deterministic region merge). This mirrors ``_parse_manifest``, which
        already resolves members per manifest entry.
        """
        if model == "gefs":
            # An explicit --member list is used verbatim; when absent, the full
            # perturbation set gep01..gep30 is the CLI default.
            return global_members or tuple(range(1, 31))
        return ()

    return [
        RunSpec(
            model=model,
            cycle_date=cycle_date,
            cycle_hour=cycle_hour,
            lead_time_hours=leads,
            members=_run_members(model),
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
            raise ValueError(f"Manifest model {entry['model']!r} is not supported.")
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
                store=entry.get("store"),
                allow_custom_store=bool(entry.get("allow_custom_store", False)),
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
        "--model",
        nargs="+",
        choices=SUPPORTED_MODELS,
        help="NOAA model identifier(s) (gfs or gefs); one or more. Required "
        "unless --manifest is given.",
    )
    ingest.add_argument(
        "--cycle-date",
        nargs="+",
        type=date.fromisoformat,
        help="UTC run date(s) (ISO format, e.g. 2026-07-21); one or more. "
        "Required unless --manifest is given.",
    )
    ingest.add_argument(
        "--cycle-hour",
        nargs="+",
        type=int,
        choices=(0, 6, 12, 18),
        help="UTC run cycle hour(s); one or more. Required unless --manifest "
        "is given.",
    )
    ingest.add_argument(
        "--lead-time-hours",
        nargs="+",
        type=int,
        help="Forecast lead time offset(s) from cycle time (0-384); one or "
        "more. Required unless --manifest is given.",
    )
    ingest.add_argument(
        "--member",
        nargs="+",
        type=int,
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
        "--download-dir",
        default="downloads",
        help="Local directory the GRIB2 file is downloaded to (created if " "missing).",
    )
    ingest.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Retain downloaded .grib2 and .idx files after successful ingestion.",
    )
    ingest.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live terminal progress UI and emit plain-text logs only.",
    )
    ingest.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum number of forecast files fetched/ingested concurrently "
        "per run (default 4). Bounded so NOMADS is not flooded and disk "
        "staging stays bounded.",
    )
    ingest.add_argument("--center-id", default="noaa")
    ingest.add_argument("--version-string", default="v1.0")
    ingest.add_argument("--grid-id", default="global_025deg")
    ingest.add_argument(
        "--variable",
        action="append",
        default=None,
        type=_parse_variable,
        metavar="CODE:NAME:UNIT[:SOURCE]",
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
    try:
        for spec in run_specs:
            try:
                _ingest_one_run(spec, args)
            except (CycleStoreMismatchError, LeadTimeMismatchError, Exception) as exc:  # noqa: BLE001 - report every run failure
                failed += 1
                print(f"run FAILED: model={spec.model} cycle={spec.cycle_time}: {exc}")
    finally:
        from ingestion.core.s3 import shutdown_s3_fs

        shutdown_s3_fs()
    if failed:
        print(f"{failed}/{len(run_specs)} run(s) failed.")
        return 1
    print(f"Ingested {len(run_specs)} run(s) successfully.")
    return 0


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
    # Lead-major ordering for ensemble models enables early progressive publication per settled lead.
    if spec.model != "gefs":
        items: list[tuple[int | None, int]] = [
            (None, lead) for lead in sorted(spec.lead_time_hours)
        ]
    else:
        items = [
            (member, lead)
            for lead in sorted(spec.lead_time_hours)
            for member in sorted(spec.members)
        ]

    status = asyncio.run(
        _run_wave(
            spec=spec,
            args=args,
            catalog_spec=catalog_spec,
            store_path=store_path,
            concurrency=concurrency,
            failures=failures,
        )
    )

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


async def _run_wave(
    spec: RunSpec,
    args: argparse.Namespace,
    catalog_spec: RunCatalogSpec,
    store_path: str,
    concurrency: int,
    failures: list[str],
) -> str:
    """Download and ingest every lead/member of a single forecast run.

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
            (None, lead) for lead in sorted(spec.lead_time_hours)
        ]
    else:
        items = [
            (member, lead)
            for lead in sorted(spec.lead_time_hours)
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
    cancel_event = threading.Event()
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
        pre_conn = engine.connect()
        try:
            tracker.set_init_phase("initialize_run_store")
            coordinator.initialize_run_store(
                pre_conn,
                seed_dataset=seed_dataset,
                expected_leads=spec.lead_time_hours,
                expected_members=spec.members,
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
                cancel_event=cancel_event,
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
                    expected_leads=spec.lead_time_hours,
                    expected_members=spec.members,
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
            lead_val: set(expected_members_for_lead) for lead_val in spec.lead_time_hours
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
            if cancel_event.is_set():
                return
            async with write_sem:
                if cancel_event.is_set():
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
                        cancel_event.set()
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
            if cancel_event.is_set():
                return
            dest = _destination_for(spec, staging_dir, lead=lead, member=member)

            # Stage 1: Pipeline admission (bounds total in-flight active/queued work)
            async with staging_sem:
                if cancel_event.is_set():
                    return

                # Stage 2: Bounded download
                async with download_sem:
                    if cancel_event.is_set():
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
                        if cancel_event.is_set():
                            return

                # Stage 3: Bounded decode (ProcessPool execution + parent normalization)
                # ZERO DB connections checked out during this compute-intensive phase.
                if cancel_event.is_set():
                    return
                ds: xr.Dataset | None = None
                async with decode_sem:
                    if cancel_event.is_set():
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
                if cancel_event.is_set():
                    return
                region_id = _region_id_for(lead, member)
                generation = generation_by_region.get(region_id)
                if generation is None:
                    failures.append(
                        f"{spec.model} member={member} lead={lead}: no generation for region {region_id}"
                    )
                    return

                async with write_sem:
                    if cancel_event.is_set():
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
                            cancel_event.set()
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
        results, cancelled = await await_all_workers_non_abandoning(
            pipeline_tasks, cancel_event
        )
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

    # 7. Coalesced finalization (after all worker Futures drained).
    tracker.on_finalize_start()
    t_fin_start = time.monotonic()
    fin_conn = engine.connect()
    try:
        run_id = _resolve_run_id(catalog_spec, store_path)
        finalize_result = coordinator.finalize_run(
            fin_conn,
            run_id=run_id,
            spec=catalog_spec,
            expected_leads=spec.lead_time_hours,
            expected_members=spec.members,
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
        engine.dispose()
        executor.shutdown(wait=True)
        from ingestion.core.s3 import close_wave_data_s3_fs

        close_wave_data_s3_fs()
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


#: Injectable engine factory for the CLI's catalog access. Production returns
#: the configured ingestion engine; tests replace this with an in-memory
#: SQLite engine so the CLI coordinator path can be exercised without PG.
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
    args: argparse.Namespace,
    store_path: str,
) -> RunCatalogSpec:
    """Build the run's catalog metadata from a run spec + CLI defaults."""
    center_name, center_country = _CENTER_METADATA[args.center_id]
    model_name, is_ensemble = _MODEL_METADATA[spec.model]
    grid_name, grid_resolution_km = _GRID_METADATA.get(
        args.grid_id, (args.grid_id, 0.0)
    )
    variables = tuple(args.variable) if args.variable is not None else DEFAULT_VARIABLES
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
