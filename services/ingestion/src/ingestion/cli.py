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
from datetime import date
from pathlib import Path
from typing import Any

from ingestion.core.base import (
    CycleStoreMismatchError,
    LeadTimeMismatchError,
)
from ingestion.core.catalog import VariableSpec
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

    return [
        RunSpec(
            model=model,
            cycle_date=cycle_date,
            cycle_hour=cycle_hour,
            target_lead_time_hours=leads,
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
                target_lead_time_hours=leads,
                members=members,
                store=entry.get("store"),
                allow_custom_store=bool(entry.get("allow_custom_store", False)),
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
                f"leads={sorted(spec.target_lead_time_hours)} -> {store_path}"
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
    concurrency = max(1, int(getattr(args, "concurrency", 4)))

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


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = _build_parser().parse_args(argv)
    if args.command == "ingest":
        return _run_ingest(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
