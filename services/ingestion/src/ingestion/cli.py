"""Command-line production entrypoint for the ingestion worker.

Runs the full ingestion flow for one NOAA forecast file:

    download GRIB2 (NOMADS)  ->  parse GRIB2  ->  write Zarr  ->  record catalog (ready)

This is the missing production call path that connects the ingestion library
modules: the download is performed by :class:`NOAAConnector`, then
:func:`ingestion.core.pipeline.ingest_grib_file` parses, writes Zarr, and
records the run in the PostgreSQL catalog (marked ``ready``) so the API
serving tier can discover and serve it.

The ``--store`` argument is the store of the run's *cycle*: one ``model_runs``
row represents a full forecast cycle and its store accumulates every lead.
NOMADS serves one GRIB2 file per lead, so invoke this CLI once per lead of the
cycle, passing the **same** ``--store`` each time; each lead is merged into
that store.

Usage:

    python -m ingestion.cli ingest --model gfs --cycle-date 2026-07-21 \\
        --cycle-hour 0 --lead-time-hours 6 \\
        --store s3://weather-data/gfs/2026-07-21/00/cycle.zarr \\
        --download-dir downloads

Or via the installed console script (see ``services/ingestion/pyproject.toml``):

    weather-ingest ingest --model gefs --cycle-date 2026-07-21 \\
        --cycle-hour 0 --lead-time-hours 6 --store s3://.../cycle.zarr
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime, timezone
from pathlib import Path

from ingestion.core.catalog import RunCatalogSpec, VariableSpec
from ingestion.core.pipeline import ingest_grib_file
from ingestion.providers.noaa.connector import NOAAConnector

#: Supported NOAA model identifiers (NOMADS GFS/GEFS).
SUPPORTED_MODELS = ("gfs", "gefs")

#: Default platform surface-variable mapping for NOAA GFS/GEFS files. Each
#: entry maps the raw GRIB2 ``shortName`` (``source_code``) to the platform
#: ``code`` recorded in ``forecast_variables``.
DEFAULT_VARIABLES: tuple[VariableSpec, ...] = (
    VariableSpec(
        code="temperature_2m",
        name="2-Meter Temperature",
        unit="°C",
        source_code="t",
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
            "temperature_2m:2-Meter Temperature:°C:t"
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
        "ingest", help="download, parse, and record one forecast run"
    )
    ingest.add_argument(
        "--model", required=True, choices=SUPPORTED_MODELS,
        help="NOAA model identifier (gfs or gefs).",
    )
    ingest.add_argument(
        "--cycle-date", required=True, type=date.fromisoformat,
        help="UTC run date (ISO format, e.g. 2026-07-21).",
    )
    ingest.add_argument(
        "--cycle-hour", required=True, type=int, choices=(0, 6, 12, 18),
        help="UTC run cycle hour.",
    )
    ingest.add_argument(
        "--lead-time-hours", required=True, type=int,
        help="Forecast lead time offset from cycle time (0-384).",
    )
    ingest.add_argument(
        "--store", required=True,
        help="Zarr store path/URL of the run's cycle. All leads of a cycle "
             "are merged into this store, so pass the same --store for every "
             "lead (e.g. s3://weather-data/gfs/2026-07-21/00/cycle.zarr).",
    )
    ingest.add_argument(
        "--download-dir", default="downloads",
        help="Local directory the GRIB2 file is downloaded to (created if "
             "missing).",
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


def _build_spec(args: argparse.Namespace) -> RunCatalogSpec:
    """Build the run's catalog metadata from CLI arguments."""
    center_name, center_country = _CENTER_METADATA[args.center_id]
    model_name, is_ensemble = _MODEL_METADATA[args.model]
    grid_name, grid_resolution_km = _GRID_METADATA.get(
        args.grid_id, (args.grid_id, 0.0)
    )
    cycle_time = datetime(
        args.cycle_date.year,
        args.cycle_date.month,
        args.cycle_date.day,
        args.cycle_hour,
        tzinfo=timezone.utc,
    )
    variables = (
        tuple(args.variable) if args.variable is not None else DEFAULT_VARIABLES
    )
    return RunCatalogSpec(
        center_id=args.center_id,
        center_name=center_name,
        center_country=center_country,
        model_id=args.model,
        model_name=model_name,
        is_ensemble=is_ensemble,
        resolution_km=grid_resolution_km,
        version_string=args.version_string,
        cycle_time=cycle_time,
        grid_id=args.grid_id,
        grid_name=grid_name,
        grid_resolution_km=grid_resolution_km,
        product_type="surface",
        zarr_store_path=args.store,
        variables=variables,
    )


def _run_ingest(args: argparse.Namespace) -> int:
    """Download one GRIB2 file, then run the parse/write/catalog pipeline."""
    spec = _build_spec(args)
    destination = Path(args.download_dir) / (
        f"{args.model}.t{args.cycle_hour:02d}z.pgrb2.0p25."
        f"f{args.lead_time_hours:03d}.grib2"
    )

    async def _download_and_ingest() -> None:
        async with NOAAConnector() as connector:
            await connector.download(
                args.model,
                args.cycle_date,
                args.cycle_hour,
                args.lead_time_hours,
                destination,
            )
        record = ingest_grib_file(spec, destination, args.store)
        print(
            f"Ingested run {record.id} ({record.status}) -> {args.store}"
        )

    asyncio.run(_download_and_ingest())
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = _build_parser().parse_args(argv)
    if args.command == "ingest":
        return _run_ingest(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
