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

Wave targets vs canonical cycle horizon (Phase 5B): the requested
``--lead-time-hours`` are the *wave targets* of this invocation
(``RunSpec.target_lead_time_hours``) — the leads this invocation ingests.
They are deliberately independent of the cycle's *expected horizon*: the
store's lead axis is pre-allocated with the model's canonical horizon
(``domain.horizon``, 0–240 h at 3-hour cadence) and run-level readiness is
evaluated against that horizon, so repeated disjoint-target invocations
accumulate safely into one cycle store and the run becomes ``ready`` only
when the whole horizon is committed.

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
import logging
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from ingestion.core.base import (
    CycleStoreMismatchError,
    LeadTimeMismatchError,
)
from ingestion.core.catalog import VariableSpec
from ingestion.core.config import settings
from ingestion.core.wave_runner import (  # noqa: F401 - re-exported runner API
    DEFAULT_VARIABLES,
    ConcurrencyPlan,
    RunSpec,
    _build_spec,
    _catalog_session,
    _catalog_session_factory,
    _cleanup_source,
    _cleanup_sources,
    _decode_and_normalize,
    _destination_for,
    _detect_effective_cpus,
    _new_generation,
    _region_id_for,
    _resolve_concurrency_plan,
    _resolve_run_id,
    _run_wave,
    _synthetic_spec_dataset,
)
from ingestion.providers.noaa.connector import NOAAConnector  # noqa: F401 - re-export

logger = logging.getLogger(__name__)

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
    ``_run_wave`` remains the authoritative cross-cycle protection.

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
        already resolves members per manifest entry. The member list is the
        invocation's wave targets; the store's member axis is always
        pre-allocated with the full gep01..gep30 contract (see ``_build_spec``).
        """
        if model == "gefs":
            # An explicit --member list is used verbatim; when absent, the full
            # perturbation set gep01..gep30 is the CLI default.
            return global_members or tuple(range(1, 31))
        return ()

    def _resolve_include_mean(model: str) -> bool:
        if model != "gefs":
            return False
        cli_flag = getattr(args, "include_mean", None)
        if cli_flag is not None:
            return bool(cli_flag)
        # By default, GEFS normal batch ingestion includes geavg unless the user
        # explicitly restricted the run to specific members via --member.
        return not bool(global_members)

    return [
        RunSpec(
            model=model,
            cycle_date=cycle_date,
            cycle_hour=cycle_hour,
            target_lead_time_hours=leads,
            members=_run_members(model),
            store=args.store,
            allow_custom_store=args.allow_custom_store,
            include_mean=_resolve_include_mean(model),
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
                target_lead_time_hours=leads,
                members=members,
                store=entry.get("store"),
                allow_custom_store=bool(entry.get("allow_custom_store", False)),
                include_mean=bool(entry.get("include_mean", True)),
            )
        )
    if not specs:
        raise ValueError("Manifest 'runs' list is empty.")
    return specs


def _parse_variable(spec: str) -> VariableSpec:
    """Parse a ``CODE:NAME:UNIT[:SOURCE]`` CLI variable spec."""
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) not in (3, 4) or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "variable must be CODE:NAME:UNIT[:SOURCE], e.g. "
            "temperature_2m:2-Meter Temperature:°C:t2m"
        )
    from ingestion.core.catalog import VariableSpec

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
        "more. These are this invocation's wave targets — the leads to ingest "
        "now; the cycle store's lead axis and final readiness use the model's "
        "canonical horizon (0-240 at 3-hour cadence). Required unless "
        "--manifest is given.",
    )
    ingest.add_argument(
        "--member",
        nargs="+",
        type=int,
        help="GEFS perturbation member identity/identities (1..30). For GEFS "
        "each member is downloaded as its own gepNN file and ingested "
        "independently; member identity is preserved regardless of "
        "completion order. The store's member axis is always pre-allocated "
        "with the full gep01..gep30 contract. Ignored for deterministic "
        "models.",
    )
    ingest.add_argument(
        "--include-mean",
        dest="include_mean",
        action="store_true",
        default=None,
        help="Include the official precomputed GEFS ensemble mean (geavg) when ingesting specific members.",
    )
    ingest.add_argument(
        "--no-mean",
        dest="include_mean",
        action="store_false",
        default=None,
        help="Skip ingesting the official precomputed GEFS ensemble mean (geavg). "
        "By default, GEFS batch ingestion automatically includes geavg unless --member or --no-mean is specified.",
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
        default=None,
        help="Generic concurrency shorthand: sets download, decode, and write concurrency "
        "when stage-specific flags are omitted.",
    )
    ingest.add_argument(
        "--download-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent file downloads from upstream (overrides --concurrency and WEATHER_INGEST_DOWNLOAD_CONCURRENCY).",
    )
    ingest.add_argument(
        "--decode-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent GRIB2 decode processes (overrides --concurrency and WEATHER_INGEST_DECODE_CONCURRENCY).",
    )
    ingest.add_argument(
        "--write-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent region writers (overrides --concurrency and WEATHER_INGEST_WRITE_CONCURRENCY).",
    )
    ingest.add_argument(
        "--marker-put-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent pre-update marker PUT operations (overrides WEATHER_INGEST_MARKER_PUT_CONCURRENCY).",
    )
    ingest.add_argument(
        "--marker-get-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent finalization marker GET operations (overrides WEATHER_INGEST_MARKER_GET_CONCURRENCY).",
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

    realtime = subparsers.add_parser(
        "realtime",
        help="run the realtime lead-wave scheduler (Phase 5C)",
        description="Poll upstream GFS/GEFS publication, plan bounded shared "
        "lead waves against the canonical horizon, and dispatch the existing "
        "ingestion wave runner. Requires PostgreSQL and REALTIME_ENABLED=true.",
    )
    realtime.add_argument(
        "--cycle-date",
        type=date.fromisoformat,
        default=None,
        help="Explicit cycle mode: the UTC cycle date to track (with "
        "--cycle-hour). Omit both for automatic upstream-driven selection.",
    )
    realtime.add_argument(
        "--cycle-hour",
        type=int,
        choices=(0, 6, 12, 18),
        default=None,
        help="Explicit cycle mode: the UTC cycle hour to track (with "
        "--cycle-date). Omit both for automatic upstream-driven selection.",
    )
    realtime.add_argument(
        "--once",
        action="store_true",
        help="Run exactly ONE poll iteration — discovery, planning, and wave "
        "dispatch (unless --dry-run) — then exit. Without --once the "
        "scheduler polls until shutdown.",
    )
    realtime.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and log the frontier/wave diagnostics WITHOUT dispatching "
        "any wave. Diagnostic mode; typically combined with --once.",
    )
    realtime.add_argument(
        "--download-dir",
        default="downloads",
        help="Local staging directory for wave downloads (default 'downloads').",
    )
    realtime.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Per-wave generic concurrency shorthand passed to the wave runner.",
    )
    realtime.add_argument(
        "--download-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent file downloads per wave (overrides --concurrency and WEATHER_INGEST_DOWNLOAD_CONCURRENCY).",
    )
    realtime.add_argument(
        "--decode-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent GRIB2 decode processes per wave (overrides --concurrency and WEATHER_INGEST_DECODE_CONCURRENCY).",
    )
    realtime.add_argument(
        "--write-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent region writers per wave (overrides --concurrency and WEATHER_INGEST_WRITE_CONCURRENCY).",
    )
    realtime.add_argument(
        "--marker-put-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent pre-update marker PUT operations per wave (overrides WEATHER_INGEST_MARKER_PUT_CONCURRENCY).",
    )
    realtime.add_argument(
        "--marker-get-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent finalization marker GET operations per wave (overrides WEATHER_INGEST_MARKER_GET_CONCURRENCY).",
    )
    realtime.add_argument(
        "--metrics-host",
        type=str,
        default="127.0.0.1",
        help="Bind address for the in-process Prometheus metrics endpoint (default: 127.0.0.1; use 0.0.0.0 only so a local Docker-based Prometheus can reach it).",
    )
    realtime.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Serve this process's live pipeline metrics (stage latencies, throughput, storage operations) on the given port, e.g. 9113. Disabled by default.",
    )

    gc = subparsers.add_parser(
        "gc",
        help="run the garbage collection and storage reclamation engine (Phase 6D)",
        description="Reconcile retired cycles, derive R2 eligibility, and delete "
        "retired forecast stores safely and sequentially.",
    )
    gc.add_argument(
        "--once",
        action="store_true",
        help="Execute exactly ONE reconciliation and deletion pass, then exit.",
    )
    gc.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and log GC diagnostics WITHOUT taking exclusive locks, "
        "deleting S3 stores, or mutating catalog/lifecycle metadata.",
    )
    gc.add_argument(
        "--interval-seconds",
        type=float,
        default=1800.0,
        help="Sleep interval in seconds between reconciliation passes in daemon mode "
        "(default 1800.0).",
    )
    gc.add_argument(
        "--lock-timeout-seconds",
        type=float,
        default=5.0,
        help="Maximum timeout in seconds to wait for exclusive store gate before skipping "
        "a busy cycle (default 5.0).",
    )
    gc.add_argument(
        "--bucket",
        default="weather-data",
        help="S3/MinIO bucket name holding forecast cycle stores (default 'weather-data').",
    )
    gc.add_argument(
        "--models",
        default="gfs,gefs",
        help="Comma-separated model identifiers for the planner pass (default 'gfs,gefs').",
    )
    gc.add_argument(
        "--enable-planner",
        action="store_true",
        help="Run the reclamation planner pass (enqueue reclaimable shard targets) after "
        "each bookkeeping pass. Queue writes only — no physical side effects. Also "
        "enabled via RECLAMATION_PLANNER_ENABLED=true.",
    )
    gc.add_argument(
        "--enable-delete",
        action="store_true",
        help="Authorize the reclamation worker pass to physically delete enqueued shard "
        "targets each pass (V3 planner -> worker -> bookkeeping mainline). Dangerous: "
        "also enabled via RECLAMATION_DELETE_ENABLED=true. Deleting requires explicit "
        "authorization; planner-only operation is the safe staging mode.",
    )
    gc.add_argument(
        "--enable-sweeper",
        action="store_true",
        help="Run the M3 14-day detailed metadata retention sweeper pass after each "
        "bookkeeping pass (daemon mode). Also enabled via RECLAMATION_SWEEPER_ENABLED=true.",
    )
    gc.add_argument(
        "--sweep-metadata",
        action="store_true",
        help="Execute M3 14-day detailed metadata retention sweeper pass.",
    )
    gc.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Maximum number of cycles to process per sweeper pass (default 50).",
    )
    gc.add_argument(
        "--inventory",
        action="store_true",
        help="Run one store-catalog orphan inventory pass (architecture doc section 9) "
        "and exit. Reports physical cycle stores without catalog identity; "
        "orphans beyond the recoverability frontier are marked reapable.",
    )
    gc.add_argument(
        "--inventory-reap",
        action="store_true",
        help="With --inventory: delete orphan store prefixes beyond the recoverability "
        "frontier under an exclusive store gate (fail-closed sanity guards).",
    )
    gc.add_argument(
        "--inventory-store-root",
        default=None,
        help="Store root for --inventory (default 's3://<--bucket>').",
    )

    reclamation = subparsers.add_parser(
        "reclamation",
        help="granular physical reclamation engine (Lifecycle V3 Phase 4)",
        description="Plan, execute, or requeue granular physical shard reclamation.",
    )
    rec_sub = reclamation.add_subparsers(dest="rec_action", required=True)

    rec_plan = rec_sub.add_parser("plan", help="Plan granular reclamation candidates")
    rec_plan.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Plan without queue writes (default True)",
    )
    rec_plan.add_argument(
        "--write",
        action="store_true",
        help="Write planned targets to reclamation_queue",
    )
    rec_plan.add_argument(
        "--models",
        default="gfs,gefs",
        help="Comma-separated model identifiers (default 'gfs,gefs')",
    )

    rec_work = rec_sub.add_parser("work", help="Execute granular reclamation worker pass")
    rec_work.add_argument(
        "--delete",
        action="store_true",
        default=False,
        help="Authorize physical S3 deletion (default False)",
    )
    rec_work.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Max shard targets per batch (default 100)",
    )
    rec_work.add_argument(
        "--lease-seconds",
        type=float,
        default=60.0,
        help="Lease duration in seconds (default 60.0)",
    )

    rec_requeue = rec_sub.add_parser(
        "requeue",
        help="Recover stuck non-terminal reclamation targets (failed quarantine, "
        "stale-lease deleting, legacy stuck queued rows over missing objects)",
    )
    rec_requeue.add_argument(
        "--model-id",
        default=None,
        help="Optional model identifier to filter requeue",
    )
    rec_requeue.add_argument(
        "--run-id",
        default=None,
        help="Optional run ID to filter requeue",
    )

    # -------------------------------------------------------------------------
    # Monitoring & Operational Health Subcommands (Runtime Monitoring)
    # -------------------------------------------------------------------------
    status_parser = subparsers.add_parser(
        "status",
        help="print runtime platform health summary and operational metrics (TASK 20)",
        description="Operator-facing runtime health summary covering PostgreSQL, MinIO, "
        "GFS/GEFS ingestion, Lifecycle, Reclamation, and active alerts.",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output health summary in JSON format",
    )
    status_parser.add_argument(
        "--download-dir",
        default="downloads",
        help="Path to temporary download staging directory to inspect capacity",
    )

    subparsers.add_parser(
        "diagnostics",
        help="print deep operational diagnostics and stage latency breakdown",
        description="Deep operational diagnostics covering process internals, PostgreSQL "
        "catalog tables, lock waits, recent pipeline milestone timings, and alert history.",
    )

    subparsers.add_parser(
        "audit",
        help="run deep lifecycle contract, invariant, and anti-resurrection audit",
        description="Audit permanent anti-resurrection tombstones, lifecycle state transitions, "
        "and GEFS perturbation member completeness.",
    )

    subparsers.add_parser(
        "alert-check",
        help="evaluate alert rules and dispatch notifications to configured sinks",
        description="Evaluates all monitoring rules (resources, database, storage, ingestion, "
        "lifecycle). Dispatches to logs and optional webhook. Exits with code 0 (healthy/info), "
        "1 (warning), or 2 (critical).",
    )

    metrics_parser = subparsers.add_parser(
        "metrics",
        help="run the Prometheus metrics exporter (scrape-time collection)",
        description=(
            "Runs the ingestion Prometheus exporter. On every HTTP scrape the "
            "platform collectors re-execute (probe-style, fail-open per "
            "collector) and a fresh exposition is returned. Binds 127.0.0.1 by "
            "default; use --host 0.0.0.0 only so a local Docker-based "
            "Prometheus can reach this host exporter."
        ),
    )
    metrics_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1; keep metrics off public interfaces)",
    )
    metrics_parser.add_argument(
        "--port",
        type=int,
        default=9112,
        help="HTTP port for the Prometheus scraper server daemon (default: 9112)",
    )
    metrics_parser.add_argument(
        "--print",
        action="store_true",
        help="collect once, print the exposition to stdout, and exit without starting the HTTP server",
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
                f"leads={sorted(spec.target_lead_time_hours)} -> {store_path}"
            )
        return 0

    from ingestion.core.s3 import verify_object_store_preflight

    verify_object_store_preflight(settings)

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


def _ingest_one_run(spec: RunSpec, args: argparse.Namespace) -> None:
    """Download and ingest every lead/member target of a single forecast run.

    Delegates the wave execution to the reusable runner
    (``ingestion.core.wave_runner._run_wave``), which implements the approved
    region-write concurrency protocol:

    * retained-seed fresh-store initialization (pre-allocated with the
      canonical cycle horizon);
    * one wave-level EXCLUSIVE pre-update (run -> partial + UPDATING markers);
    * bounded region workers (SHARED gate + region locks + generation check);
    * one coalesced finalization after the wave drains (readiness evaluated
      against the canonical cycle horizon).

    A failure in one file does not abort the others; the wave finalizer still
    runs (the run stays partial if any target is incomplete).

    Args:
        spec: The forecast-run specification (wave targets).
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
    req_concurrency = getattr(args, "concurrency", None)
    if req_concurrency is not None:
        req_concurrency = max(1, int(req_concurrency))

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

    failures: list[str] = []
    status = asyncio.run(
        _run_wave(
            spec=spec,
            args=args,
            catalog_spec=catalog_spec,
            store_path=store_path,
            concurrency=req_concurrency,
            failures=failures,
            download_concurrency=getattr(args, "download_concurrency", None),
            decode_concurrency=getattr(args, "decode_concurrency", None),
            write_concurrency=getattr(args, "write_concurrency", None),
            marker_put_concurrency=getattr(args, "marker_put_concurrency", None),
            marker_get_concurrency=getattr(args, "marker_get_concurrency", None),
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


def _run_realtime(args: argparse.Namespace) -> int:
    """Run the realtime lead-wave scheduler (Phase 5C).

    Semantics: ``--once`` performs exactly one poll iteration (discovery,
    planning, and wave dispatch unless ``--dry-run``) and exits;
    ``--dry-run`` plans and logs diagnostics without dispatching. Without
    ``--once`` the scheduler polls until shutdown (SIGINT/SIGTERM stop
    promptly; an active wave drains non-abandoningly via the wave runner).

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code.
    """
    if (args.cycle_date is None) != (args.cycle_hour is None):
        print(
            "--cycle-date and --cycle-hour must be given together (explicit "
            "cycle mode) or both omitted (automatic upstream-driven selection)."
        )
        return 2
    if not settings.REALTIME_ENABLED:
        print(
            "Realtime ingestion is disabled: set REALTIME_ENABLED=true "
            "(see .env.example). Big-batch `ingest` is unaffected."
        )
        return 2

    import signal

    from ingestion.core.db import engine as catalog_engine
    from ingestion.realtime.leadership import SchedulerLeadership
    from ingestion.realtime.scheduler import CycleIdentity, RealtimeScheduler

    cycle_override = None
    if args.cycle_date is not None and args.cycle_hour is not None:
        cycle_override = CycleIdentity(
            cycle_date=args.cycle_date, cycle_hour=args.cycle_hour
        )
    req_concurrency = getattr(args, "concurrency", None)
    if req_concurrency is not None:
        req_concurrency = max(1, int(req_concurrency))
    scheduler = RealtimeScheduler(
        conn_settings=settings,
        leadership=SchedulerLeadership(catalog_engine),
        cycle_override=cycle_override,
        download_dir=args.download_dir,
        concurrency=req_concurrency,
        download_concurrency=getattr(args, "download_concurrency", None),
        decode_concurrency=getattr(args, "decode_concurrency", None),
        write_concurrency=getattr(args, "write_concurrency", None),
        marker_put_concurrency=getattr(args, "marker_put_concurrency", None),
        marker_get_concurrency=getattr(args, "marker_get_concurrency", None),
    )

    def _handle_signal(signum: int, frame: object) -> None:
        del signum, frame
        scheduler.request_stop()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handle_signal)
            except (ValueError, OSError):  # not the main thread / unsupported
                pass

    metrics_port = getattr(args, "metrics_port", None)
    if metrics_port:
        # Serve THIS process's live pipeline counters (stage latencies,
        # throughput, storage operations) from a background HTTP thread.
        # The registry is already maintained by the wave runner; the server
        # adds no probing and no persistent state.
        import threading

        from ingestion.monitoring.exporter import serve_live_registry

        threading.Thread(
            target=serve_live_registry,
            kwargs={"host": getattr(args, "metrics_host", "127.0.0.1"), "port": metrics_port},
            daemon=True,
        ).start()

    return scheduler.run(once=args.once, dry_run=args.dry_run)


def _gc_pipeline_pass(
    catalog_engine: Engine,
    *,
    models: list[str],
    enable_planner: bool,
    enable_delete: bool,
    enable_sweeper: bool,
    batch_size: int,
    dry_run: bool = False,
) -> str:
    """Execute one full V3 GC pipeline pass: bookkeeping -> planner -> worker -> sweeper.

    This is the automated wiring of the ``planner -> worker -> bookkeeping``
    mainline: the gc daemon (or a cron-invoked ``--once`` run) drives every
    stage, so expired units are actually reclaimed without an operator running
    ``reclamation plan --write`` / ``reclamation work --delete`` by hand.

    Staging model (two-level authorization):
    * ``enable_planner`` — queue writes only, no physical side effects. Safe
      staging mode.
    * ``enable_delete`` — authorizes the worker stage to physically delete
      enqueued shard targets.

    Each stage is failure-isolated: a stage error is logged and the pass
    continues with the next stage so a transient DB/S3 fault cannot kill the
    daemon loop.

    Returns a one-line structured summary of all stage outcomes.
    """
    from ingestion.core.db import SessionLocal
    from ingestion.gc.finalizer import run_lifecycle_bookkeeping_pass
    from ingestion.gc.planner import plan_reclamation_pass
    from ingestion.gc.sweeper import run_metadata_sweeper_pass
    from ingestion.gc.worker import run_reclamation_worker_pass

    parts: list[str] = []

    try:
        bk = run_lifecycle_bookkeeping_pass(catalog_engine, dry_run=dry_run)
        parts.append(
            f"bookkeeping claimed={len(bk.claimed_cycles)} "
            f"finalized={len(bk.finalized_cycles)} "
            f"blocked={len(bk.blocked_cycles)} failed={len(bk.failed_cycles)}"
        )
    except Exception as exc:
        logger.exception("GC bookkeeping stage failed")
        parts.append(f"bookkeeping=ERROR({type(exc).__name__})")

    if enable_planner:
        try:
            with SessionLocal() as session:
                plan = plan_reclamation_pass(session, models=models, dry_run=dry_run)
            parts.append(
                f"planner enqueued={plan.enqueued_count} "
                f"reclaimable={plan.reclaimable_shards}"
            )
        except Exception as exc:
            logger.exception("GC planner stage failed")
            parts.append(f"planner=ERROR({type(exc).__name__})")

    if enable_delete and not dry_run:
        try:
            with SessionLocal() as session:
                w = run_reclamation_worker_pass(session, delete_enabled=True, batch_size=batch_size)
            parts.append(
                f"worker claimed={w.claimed_count} deleted={w.deleted_count} "
                f"revalidated={w.revalidated_held_count} failed={w.failed_count} "
                f"markers={w.markers_cleaned_count}"
            )
        except Exception as exc:
            logger.exception("GC worker stage failed")
            parts.append(f"worker=ERROR({type(exc).__name__})")

    if enable_sweeper:
        try:
            sw = run_metadata_sweeper_pass(catalog_engine, dry_run=dry_run, batch_size=batch_size)
            parts.append(
                f"sweeper swept={len(sw.swept_cycles)} "
                f"failed={len(sw.failed_cycles)} runs_deleted={sw.total_model_runs_deleted}"
            )
        except Exception as exc:
            logger.exception("GC sweeper stage failed")
            parts.append(f"sweeper=ERROR({type(exc).__name__})")

    return "; ".join(parts)


def _run_gc(args: argparse.Namespace) -> int:
    """Run the garbage collection (GC) and storage reclamation engine (Phase 6D).

    Supported modes (Lifecycle V3 converged model: physical deletion happens
    exclusively through the reclamation planner + worker at
    (variable, valid_time) granularity; bookkeeping itself performs zero
    physical storage operations):
    * ``--sweep-metadata``: Execute M3 14-day detailed metadata retention sweeper pass.
    * ``--inventory``: Run one store-catalog orphan inventory pass and exit.
    * ``--once --dry-run``: Plan and log diagnostics without mutating PostgreSQL.
    * ``--once``: Acquire GC leadership, execute one pipeline pass, and exit.
    * Daemon (default): Acquire GC leadership and loop: pipeline pass -> sleep -> pipeline pass.

    The pipeline pass is bookkeeping plus the automatically-wired V3 mainline
    (staged by two-level authorization):
    * ``--enable-planner`` (or RECLAMATION_PLANNER_ENABLED): enqueue reclaimable
      shard targets after each bookkeeping pass (queue writes only).
    * ``--enable-delete`` (or RECLAMATION_DELETE_ENABLED): authorize the worker
      stage to physically delete enqueued shard targets.
    * ``--enable-sweeper`` (or RECLAMATION_SWEEPER_ENABLED): include the 14-day
      metadata sweeper pass in each pipeline pass.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit code (0 on success, non-zero on fatal leadership/runtime error).
    """
    import signal
    import time
    from ingestion.core.db import engine as catalog_engine
    from ingestion.gc.leadership import GcLeadership
    from ingestion.gc.finalizer import run_lifecycle_bookkeeping_pass

    if getattr(args, "inventory", False):
        from ingestion.gc.inventory import run_orphan_inventory

        bucket = str(args.bucket)
        store_root = getattr(args, "inventory_store_root", None) or f"s3://{bucket}"
        inv = run_orphan_inventory(
            catalog_engine,
            store_root=store_root,
            reap=bool(args.inventory_reap),
            timeout_seconds=float(args.lock_timeout_seconds),
        )
        print(
            f"Orphan Inventory: stores={inv.discovered_stores} "
            f"cataloged_cycles={inv.cataloged_cycles} "
            f"orphans={len(inv.orphan_stores)} "
            f"beyond_frontier={sum(1 for o in inv.orphan_stores if o.beyond_frontier)} "
            f"reaped={len(inv.reaped_stores)} blocked={len(inv.reap_blocked)}"
        )
        for orphan in inv.orphan_stores:
            flag = "REAPABLE" if orphan.beyond_frontier else "recoverable-frontier"
            print(f"  ORPHAN [{flag}] {orphan.store_path} cycle={orphan.cycle_time.isoformat()}")
        for path in inv.reap_blocked:
            print(f"  REAP-BLOCKED {path}")
        return 0

    if getattr(args, "sweep_metadata", False):
        from ingestion.gc.sweeper import run_metadata_sweeper_pass

        b_size = int(getattr(args, "batch_size", 50))
        res = run_metadata_sweeper_pass(
            catalog_engine,
            dry_run=bool(args.dry_run),
            batch_size=b_size,
        )
        print(
            f"Metadata Sweeper Pass: dry_run={res.dry_run}, "
            f"candidates={len(res.candidates)}, "
            f"swept={len(res.swept_cycles)}, "
            f"failed={len(res.failed_cycles)}, "
            f"model_runs_deleted={res.total_model_runs_deleted}"
        )
        return 0 if len(res.failed_cycles) == 0 else 1

    dry_run = bool(args.dry_run)
    interval = max(1.0, float(args.interval_seconds))

    from ingestion.core.config import settings as ingest_settings

    enable_planner = bool(args.enable_planner) or bool(ingest_settings.RECLAMATION_PLANNER_ENABLED)
    enable_delete = bool(args.enable_delete) or bool(ingest_settings.RECLAMATION_DELETE_ENABLED)
    enable_sweeper = bool(args.enable_sweeper) or bool(ingest_settings.RECLAMATION_SWEEPER_ENABLED)
    models = [m.strip().lower() for m in str(args.models).split(",") if m.strip()]

    if dry_run:
        # Dry-run performs zero mutations and does not acquire destructive leadership
        if enable_planner:
            # The planner supports a true dry-run (log-only candidate discovery);
            # the worker stage is skipped entirely in dry-run mode.
            print(
                "GC dry-run pass: "
                + _gc_pipeline_pass(
                    catalog_engine,
                    models=models,
                    enable_planner=True,
                    enable_delete=False,
                    enable_sweeper=enable_sweeper,
                    batch_size=int(getattr(args, "batch_size", 50)),
                    dry_run=True,
                )
            )
        else:
            run_lifecycle_bookkeeping_pass(
                catalog_engine,
                dry_run=True,
            )
        return 0

    logger.info(
        "GC daemon mode: planner=%s delete=%s sweeper=%s interval=%ss models=%s",
        enable_planner,
        enable_delete,
        enable_sweeper,
        interval,
        models,
    )
    if enable_delete:
        logger.warning(
            "GC PHYSICAL DELETION IS AUTHORIZED (planner -> worker mainline active); "
            "shard targets enqueued by the planner will be permanently deleted."
        )

    leadership = GcLeadership(catalog_engine)
    if not leadership.acquire():
        logger.warning("Another GC leader is currently active; exiting.")
        print("Another GC leader process is currently running. Exiting.")
        return 0 if args.once else 1

    stop_requested = False

    def _handle_signal(signum: int, frame: object) -> None:
        nonlocal stop_requested
        del signum, frame
        stop_requested = True

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handle_signal)
            except (ValueError, OSError):
                pass

    try:
        while not stop_requested:
            if not leadership.check_leadership():
                logger.error("GC leadership lost; attempting reacquisition...")
                if not leadership.acquire():
                    logger.error("Failed to reacquire GC leadership; exiting.")
                    return 1

            summary = _gc_pipeline_pass(
                catalog_engine,
                models=models,
                enable_planner=enable_planner,
                enable_delete=enable_delete,
                enable_sweeper=enable_sweeper,
                batch_size=int(getattr(args, "batch_size", 50)),
                dry_run=False,
            )
            print(f"GC pass: {summary}", flush=True)

            if args.once or stop_requested:
                break

            # Bounded sleep with signal responsiveness
            sleep_end = time.monotonic() + interval
            while time.monotonic() < sleep_end and not stop_requested:
                time.sleep(min(1.0, sleep_end - time.monotonic()))
        return 0
    finally:
        leadership.release()


def _run_reclamation(args: argparse.Namespace) -> int:
    """Execute granular reclamation CLI action (plan, work, or requeue)."""
    from ingestion.core.db import SessionLocal
    from ingestion.gc.planner import plan_reclamation_pass
    from ingestion.gc.worker import (
        requeue_failed_reclamation_targets,
        run_reclamation_worker_pass,
    )

    action = getattr(args, "rec_action", None)
    with SessionLocal() as session:
        if action == "plan":
            dry_run = not bool(getattr(args, "write", False))
            models = [m.strip().lower() for m in str(args.models).split(",") if m.strip()]
            res = plan_reclamation_pass(session, models=models, dry_run=dry_run)
            print(
                f"Reclamation Plan: dry_run={res.dry_run}, "
                f"total_committed={res.total_committed_shards}, "
                f"active_held={res.active_held_shards}, "
                f"reclaimable={res.reclaimable_shards}, "
                f"enqueued={res.enqueued_count}"
            )
            return 0
        if action == "work":
            del_en = bool(getattr(args, "delete", False))
            b_size = int(getattr(args, "batch_size", 100))
            l_secs = float(getattr(args, "lease_seconds", 60.0))
            w_res = run_reclamation_worker_pass(
                session,
                batch_size=b_size,
                lease_seconds=l_secs,
                delete_enabled=del_en,
            )
            print(
                f"Reclamation Worker: claimed={w_res.claimed_count}, "
                f"deleted={w_res.deleted_count}, "
                f"revalidated={w_res.revalidated_held_count}, "
                f"failed={w_res.failed_count}, "
                f"markers_cleaned={w_res.markers_cleaned_count}"
            )
            return 0
        if action == "requeue":
            m_id = getattr(args, "model_id", None)
            r_id = getattr(args, "run_id", None)
            count = requeue_failed_reclamation_targets(session, model_id=m_id, run_id=r_id)
            print(f"Recovered {count} stuck non-terminal reclamation targets.")
            return 0
    print(f"Unknown reclamation action: {action}")
    return 2


def _run_status(args: argparse.Namespace) -> int:
    import json
    from dataclasses import asdict
    from ingestion.core.config import settings
    from ingestion.core.db import engine
    from ingestion.monitoring import (
        ALERT_ENGINE,
        INGESTION_COLLECTOR,
        LEAK_DETECTOR,
        RESOURCE_COLLECTOR,
        AlertSeverity,
        LifecycleHealthCollector,
        PostgresHealthCollector,
        StorageHealthCollector,
        render_platform_status,
    )

    dl_dir = getattr(args, "download_dir", "downloads")
    res_data = RESOURCE_COLLECTOR.collect_and_export(disk_paths=(".", dl_dir))
    pg_data = PostgresHealthCollector(engine).collect()
    storage_data = StorageHealthCollector(
        endpoint_url=getattr(settings, "MINIO_ENDPOINT", "http://localhost:9000"),
        bucket=getattr(settings, "MINIO_BUCKET_NAME", "weather-data"),
        access_key=getattr(settings, "MINIO_ACCESS_KEY", "minio_admin"),
        secret_key=getattr(settings, "MINIO_SECRET_KEY", "minio_password"),
    ).probe()
    lifecycle_data = LifecycleHealthCollector(engine).collect()
    gfs_lag = INGESTION_COLLECTOR.evaluate_lag("gfs")
    gefs_lag = INGESTION_COLLECTOR.evaluate_lag("gefs")
    gfs_state = INGESTION_COLLECTOR.get_model_state("gfs")
    gefs_state = INGESTION_COLLECTOR.get_model_state("gefs")
    leak_data = LEAK_DETECTOR.evaluate_leak()

    ingestion_data = {
        "gfs": {"lag": gfs_lag, "stuck": INGESTION_COLLECTOR.check_stuck("gfs")},
        "gefs": {"lag": gefs_lag, "stuck": INGESTION_COLLECTOR.check_stuck("gefs")},
    }

    active_alerts = ALERT_ENGINE.evaluate_rules(
        resource_data=res_data,
        postgres_data=pg_data,
        lifecycle_data=lifecycle_data,
        ingestion_data=ingestion_data,
        storage_data=storage_data,
        leak_data=leak_data,
    )

    if getattr(args, "json", False):
        summary_dict = {
            "platform_status": (
                "CRITICAL"
                if any(a.severity == AlertSeverity.CRITICAL for a in active_alerts)
                else ("DEGRADED" if active_alerts else "HEALTHY")
            ),
            "postgres": {
                "connected": pg_data.connected,
                "connections": pg_data.active_connections,
                "max_connections": pg_data.max_connections,
                "utilization_pct": pg_data.connection_utilization_pct,
                "total_size_bytes": pg_data.total_size_bytes,
            },
            "storage": {
                "connected": storage_data.connected,
                "latency_ms": storage_data.latency_ms,
            },
            "gfs": {
                "lag_cycles": gfs_lag.lag_cycles,
                "lag_hours": gfs_lag.lag_hours,
                "latest_ready": str(gfs_lag.latest_ready_cycle),
                "expected_latest": str(gfs_lag.latest_expected_cycle),
            },
            "gefs": {
                "lag_cycles": gefs_lag.lag_cycles,
                "lag_hours": gefs_lag.lag_hours,
                "latest_ready": str(gefs_lag.latest_ready_cycle),
                "expected_latest": str(gefs_lag.latest_expected_cycle),
            },
            "lifecycle": {
                "active_cycles": lifecycle_data.active_cycles,
                "claimed_cycles": lifecycle_data.claimed_cycles,
                "stuck_claims": lifecycle_data.stuck_claims_warning + lifecycle_data.stuck_claims_critical,
                "tombstones": lifecycle_data.tombstone_cycles,
            },
            "reclamation": {
                "queued": lifecycle_data.reclamation.queued_count,
                "deleting": lifecycle_data.reclamation.deleting_count,
                "deleted": lifecycle_data.reclamation.deleted_count,
                "failed": lifecycle_data.reclamation.failed_count,
            },
            "metadata_retention": {
                "eligible_backlog": lifecycle_data.sweeper_unpurged_count,
                "oldest_overdue_seconds": lifecycle_data.sweeper_oldest_overdue_s,
            },
            "active_alerts": [asdict(a) for a in active_alerts],
        }
        print(json.dumps(summary_dict, indent=2, default=str))
    else:
        text_out = render_platform_status(
            resource_data=res_data,
            postgres_data=pg_data,
            storage_data=storage_data,
            lifecycle_data=lifecycle_data,
            gfs_lag=gfs_lag,
            gefs_lag=gefs_lag,
            gfs_state=gfs_state,
            gefs_state=gefs_state,
            leak_data=leak_data,
            active_alerts=active_alerts,
        )
        print(text_out)

    return 0


def _run_diagnostics(args: argparse.Namespace) -> int:
    from ingestion.core.config import settings
    from ingestion.core.db import engine
    from ingestion.monitoring import (
        ALERT_ENGINE,
        INGESTION_COLLECTOR,
        LifecycleHealthCollector,
        PostgresHealthCollector,
        RESOURCE_COLLECTOR,
        StorageHealthCollector,
        render_diagnostics,
    )

    res_data = RESOURCE_COLLECTOR.collect_and_export()
    pg_data = PostgresHealthCollector(engine).collect()
    storage_data = StorageHealthCollector(
        endpoint_url=getattr(settings, "MINIO_ENDPOINT", "http://localhost:9000"),
        bucket=getattr(settings, "MINIO_BUCKET_NAME", "weather-data"),
        access_key=getattr(settings, "MINIO_ACCESS_KEY", "minio_admin"),
        secret_key=getattr(settings, "MINIO_SECRET_KEY", "minio_password"),
    ).probe()
    lifecycle_data = LifecycleHealthCollector(engine).collect()
    gfs_state = INGESTION_COLLECTOR.get_model_state("gfs")
    gefs_state = INGESTION_COLLECTOR.get_model_state("gefs")
    recent_events = ALERT_ENGINE.memory_sink.get_recent(limit=20)

    out = render_diagnostics(
        resource_data=res_data,
        postgres_data=pg_data,
        storage_data=storage_data,
        lifecycle_data=lifecycle_data,
        gfs_state=gfs_state,
        gefs_state=gefs_state,
        recent_events=recent_events,
    )
    print(out)
    return 0


def _run_audit(args: argparse.Namespace) -> int:
    from ingestion.core.db import engine
    from ingestion.monitoring import (
        INGESTION_COLLECTOR,
        LifecycleHealthCollector,
        render_audit,
    )

    lifecycle_data = LifecycleHealthCollector(engine).collect()
    gfs_comp = INGESTION_COLLECTOR.evaluate_gfs_completeness()
    gefs_comp = INGESTION_COLLECTOR.evaluate_gefs_completeness()
    out = render_audit(
        lifecycle_data=lifecycle_data,
        gfs_completeness=gfs_comp,
        gefs_completeness=gefs_comp,
    )
    print(out)
    return 1 if lifecycle_data.violations else 0


def _run_alert_check(args: argparse.Namespace) -> int:
    from ingestion.core.config import settings
    from ingestion.core.db import engine
    from ingestion.monitoring import (
        ALERT_ENGINE,
        AlertSeverity,
        INGESTION_COLLECTOR,
        LEAK_DETECTOR,
        LifecycleHealthCollector,
        PostgresHealthCollector,
        RESOURCE_COLLECTOR,
        StorageHealthCollector,
    )

    res_data = RESOURCE_COLLECTOR.collect_and_export()
    pg_data = PostgresHealthCollector(engine).collect()
    storage_data = StorageHealthCollector(
        endpoint_url=getattr(settings, "MINIO_ENDPOINT", "http://localhost:9000"),
        bucket=getattr(settings, "MINIO_BUCKET_NAME", "weather-data"),
        access_key=getattr(settings, "MINIO_ACCESS_KEY", "minio_admin"),
        secret_key=getattr(settings, "MINIO_SECRET_KEY", "minio_password"),
    ).probe()
    lifecycle_data = LifecycleHealthCollector(engine).collect()
    gfs_lag = INGESTION_COLLECTOR.evaluate_lag("gfs")
    gefs_lag = INGESTION_COLLECTOR.evaluate_lag("gefs")
    leak_data = LEAK_DETECTOR.evaluate_leak()

    ingestion_data = {
        "gfs": {"lag": gfs_lag, "stuck": INGESTION_COLLECTOR.check_stuck("gfs")},
        "gefs": {"lag": gefs_lag, "stuck": INGESTION_COLLECTOR.check_stuck("gefs")},
    }

    ALERT_ENGINE.evaluate_and_dispatch(
        resource_data=res_data,
        postgres_data=pg_data,
        lifecycle_data=lifecycle_data,
        ingestion_data=ingestion_data,
        storage_data=storage_data,
        leak_data=leak_data,
    )

    active = ALERT_ENGINE.deduplicator.get_active_alerts()
    crit_count = sum(1 for a in active if a.severity == AlertSeverity.CRITICAL)
    warn_count = sum(1 for a in active if a.severity == AlertSeverity.WARNING)

    if crit_count > 0:
        print(f"ALERT CHECK: CRITICAL ({crit_count} critical, {warn_count} warning alerts active)")
        for a in active:
            if a.severity == AlertSeverity.CRITICAL:
                print(f"  [CRITICAL] {a.name} ({a.scope}): {a.summary}")
        return 2
    if warn_count > 0:
        print(f"ALERT CHECK: WARNING ({warn_count} warning alerts active)")
        for a in active:
            if a.severity == AlertSeverity.WARNING:
                print(f"  [WARNING] {a.name} ({a.scope}): {a.summary}")
        return 1
    print("ALERT CHECK: OK (all systems healthy)")
    return 0


def _run_metrics(args: argparse.Namespace) -> int:
    from ingestion.monitoring.exporter import _collect_and_render, serve_metrics

    if args.print:
        print(_collect_and_render().decode("utf-8"), end="")
        return 0

    return serve_metrics(host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = _build_parser().parse_args(argv)
    if args.command == "ingest":
        return _run_ingest(args)
    if args.command == "realtime":
        return _run_realtime(args)
    if args.command == "gc":
        return _run_gc(args)
    if args.command == "reclamation":
        return _run_reclamation(args)
    if args.command == "status":
        return _run_status(args)
    if args.command == "diagnostics":
        return _run_diagnostics(args)
    if args.command == "audit":
        return _run_audit(args)
    if args.command == "alert-check":
        return _run_alert_check(args)
    if args.command == "metrics":
        return _run_metrics(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
