"""Production ingestion orchestration: parse GRIB2 -> write Zarr -> record catalog.

This is the thin runtime boundary that wires the ingestion library modules
together into a real production call path. It deliberately contains **no**
network/scheduling logic (the CLI owns download access) and **no** persistence
details (``catalog.py`` owns PostgreSQL), keeping every layer single-purpose:

* ``parse_grib2``  -> decode + normalize a GRIB2 file (``providers/noaa/parser.py``)
* ``write_dataset`` -> persist the dataset to Zarr (``core/zarr_writer.py``)
* ``record_ingested_dataset`` -> upsert the PostgreSQL catalog as a ``ready`` run
  (``core/catalog.py``)

One ``model_runs`` row represents a full forecast cycle (``UNIQUE(model_version_id,
cycle_time)`` per DATABASE.md), and its Zarr store accumulates every lead. NOMADS
serves one GRIB2 file per lead, so :func:`ingest_grib_file` merges each new lead
into the cycle store along ``lead_time_hours`` (re-ingesting a lead replaces it)
before recording the run.

The entry point :func:`ingest_grib_file` is the reusable unit a scheduler or
worker (e.g. a future Celery task) can call; the console entrypoint
``weather-ingest`` (``ingestion.cli``) adds the network download step on top.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from ingestion.core.catalog import (
    ModelRunRecord,
    RunCatalogSpec,
    VariableSpec,
    record_ingested_dataset,
)
from ingestion.core.zarr_writer import read_dataset, store_exists, write_dataset
from ingestion.providers.noaa.parser import parse_grib2


def _apply_variable_mapping(
    dataset: xr.Dataset,
    variables: tuple[VariableSpec, ...],
) -> xr.Dataset:
    """Rename a parsed dataset's data variables to the platform vocabulary.

    The GRIB2 decoder emits raw ``shortName`` data variables (e.g. ``t`` for
    2-metre temperature). Each :class:`VariableSpec` in ``variables`` may carry
    a ``source_code`` naming the raw variable it maps from; matching variables
    are renamed to the platform ``code`` so the Zarr store is self-consistent
    with the ``forecast_variables``/``forecast_products`` catalog rows.

    Args:
        dataset: The parsed dataset (raw GRIB2 variable names).
        variables: The run's :class:`VariableSpec` catalog metadata.

    Returns:
        The dataset with raw variable names renamed to platform codes. Unknown
        raw variables are left untouched.
    """
    rename: dict[str, str] = {}
    for variable in variables:
        source = variable.source_code or variable.code
        if source in dataset.data_vars:
            rename[source] = variable.code
    if not rename:
        return dataset
    return dataset.rename(rename)


def ingest_grib_file(
    spec: RunCatalogSpec,
    grib_path: str | Path,
    store_path: str,
) -> ModelRunRecord:
    """Parse a GRIB2 file, write it to a Zarr store, and record it in the catalog.

    This is the production orchestration call path for an already-downloaded
    forecast file. ``store_path`` is the store of the run's *cycle*: one
    ``model_runs`` row represents a full forecast cycle and its store
    accumulates every lead (DATABASE.md ``UNIQUE(model_version_id,
    cycle_time)``). Because a GRIB2 file holds a single lead, each call merges
    the new lead into the cycle store:

        1. ``parse_grib2`` decodes and normalizes the GRIB2 file.
        2. Data variables are renamed to the platform vocabulary (per
           ``spec.variables``).
        3. The single-lead dataset is merged into the cycle store along
           ``lead_time_hours`` (re-ingesting a lead replaces it).
        4. ``write_dataset`` persists the merged dataset to the store.
        5. ``record_ingested_dataset`` upserts the run (and its catalog rows)
           with ``status='ready'`` so the API serving tier can serve it.

    Args:
        spec: The catalog metadata of the run (model, version, cycle, grid,
            variables).
        grib_path: Path to the downloaded GRIB2 file.
        store_path: Zarr store path/URL of the run's cycle. All leads of a
            cycle share this store.

    Returns:
        The recorded :class:`ModelRunRecord` in the ``ready`` state.

    Raises:
        GribParsingError: If the GRIB2 file cannot be decoded.
        ValueError: If the Zarr store target is unsupported.
        sqlalchemy error: If the catalog write fails (no row is committed).
    """
    dataset = parse_grib2(grib_path)
    dataset = _apply_variable_mapping(dataset, spec.variables)
    dataset = _merge_lead(dataset, store_path)
    write_dataset(dataset, store_path)
    return record_ingested_dataset(spec, dataset, effective_store_path=store_path)


def _merge_lead(dataset: xr.Dataset, store_path: str) -> xr.Dataset:
    """Merge a single-lead dataset into a cycle's Zarr store.

    The parser emits ``lead_time_hours`` as a scalar coordinate (one GRIB file
    holds one lead), while a cycle store needs it as a dimension. The dataset
    is expanded first so subsequent leads can be concatenated along
    ``lead_time_hours``. When the cycle store already exists, the new lead is
    merged in and any previous copy of the same lead is replaced (re-ingest
    idempotency); the merged dataset is returned so the caller writes the
    complete cycle back to the store.

    Args:
        dataset: The normalized single-lead dataset for this GRIB file.
        store_path: The cycle store to merge into (may not exist yet).

    Returns:
        The dataset for the whole cycle: the new lead alone on a fresh ingest,
        otherwise the accumulated cycle with the new lead merged in.
    """
    if "lead_time_hours" not in dataset.dims:
        dataset = dataset.expand_dims("lead_time_hours")
    if not store_exists(store_path):
        return dataset

    existing = read_dataset(store_path)
    if "lead_time_hours" not in existing.dims:
        return dataset
    new_lead = int(dataset["lead_time_hours"].values[0])
    keep = (existing["lead_time_hours"] != new_lead).values
    merged = xr.concat(
        [existing.isel(lead_time_hours=keep), dataset],
        dim="lead_time_hours",
        coords="minimal",
    )
    return merged.sortby("lead_time_hours")
