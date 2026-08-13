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

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr

from ingestion.core.base import IngestionError, LeadTimeMismatchError
from ingestion.core.catalog import (
    ModelRunRecord,
    RunCatalogSpec,
    VariableSpec,
    record_ingested_dataset,
)
from ingestion.core.zarr_writer import (
    CorruptStoreError,
    read_dataset,
    store_exists,
    store_lock,
    store_status,
    write_dataset,
)
from ingestion.providers.noaa.parser import parse_grib2


logger = logging.getLogger(__name__)


class UnitNormalizationError(IngestionError):
    """Raised when a source GRIB unit cannot be safely canonicalized.

    A mapped variable whose source ``units`` attribute is neither already equal
    to the target canonical unit nor a known source unit for that target is
    rejected rather than silently stored under a possibly-wrong label.
    """


class MissingVariableError(IngestionError):
    """Raised when *none* of the requested platform variables are present.

    ``record_run`` creates a ``forecast_products`` row for every
    ``VariableSpec`` in ``spec.variables``. If a requested variable's source
    field was not selected by the parser, the catalog would advertise a
    product with no data in the store — silent bad data. When *some* of the
    requested variables are present, ingestion proceeds and records only the
    present subset (the parser deliberately skips fields absent from a file:
    a GEFS ``pgrb2b`` product legitimately omits ``prate``); only when the
    intersection is empty does ingestion fail fast.
    """

def _unit_token(unit: str) -> str:
    """Return a whitespace/power-symbol normalized token for a unit string.

    GRIB ``units`` strings vary in spelling across producers and eccodes
    versions (e.g. ``"kg m-2 s-1"`` vs ``"kg m**-2 s**-1"`` vs
    ``"kg m^-2 s^-1"``). Stripping spaces, ``**``, and ``^`` maps equivalent
    spellings to one token for table lookup.

    Args:
        unit: A unit string (e.g. ``"kg m-2 s-1"``).

    Returns:
        A normalized token (e.g. ``"kgm-2s-1"``).
    """
    return unit.replace(" ", "").replace("**", "").replace("^", "").lower()


#: Source GRIB unit → canonical (target) unit value transforms.
#: Keyed by the target canonical unit string (the ``VariableSpec.unit``), then
#: by the normalized source ``units`` token. Each transform maps a source-unit
#: array to the equivalent canonical-unit array. Only the currently implemented
#: platform mappings are present (ENGINEERING_CONTRACT: no speculative units).
_SOURCE_TO_CANONICAL: dict[
    str, dict[str, Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]
] = {
    # GRIB 2-metre temperature ``t`` is Kelvin; canonical is Celsius.
    "°C": {"k": lambda array: array - 273.15},
    # GFS ``prate`` is a precipitation rate in kg m-2 s-1; canonical is mm/h.
    # For liquid water 1 kg m-2 == 1 mm water-equivalent depth, so the
    # rate conversion is a pure ×3600 (s-1 → h-1). This is a rate conversion,
    # not an accumulation conversion.
    "mm/h": {"kgm-2s-1": lambda array: array * 3600.0},
}


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


def _validate_required_variables(
    dataset: xr.Dataset,
    variables: tuple[VariableSpec, ...],
) -> tuple[VariableSpec, ...]:
    """Return the requested variables present in the mapped dataset.

    Every :class:`VariableSpec` in ``variables`` becomes a
    ``forecast_products`` catalog row in :func:`record_run`. A variable whose
    source field the parser did not select (e.g. a custom ``--variable`` for
    a field such as a 10 m wind that is not among
    ``SURFACE_FIELD_FILTERS``) would otherwise be silently recorded as a
    product with no corresponding data in the Zarr store. Validation runs
    after variable mapping, looking up the platform ``code`` (the renamed
    target of the variable's ``source_code``, or the code itself when no
    source is declared) in the mapped dataset.

    Partial presence degrades gracefully: the parser deliberately skips
    fields absent from a file (a GEFS ``pgrb2b`` product legitimately omits
    ``prate``), so when only a subset of the requested variables is present
    the present subset is returned (the caller records only those products)
    and the missing ones are logged as a warning. Only when *none* of the
    requested variables is present does ingestion fail fast.

    Args:
        dataset: The mapped dataset (platform variable names).
        variables: The run's :class:`VariableSpec` catalog metadata.

    Returns:
        The subset of ``variables`` present in the mapped dataset.

    Raises:
        MissingVariableError: If none of the requested variables is present
            in the mapped dataset, listing the missing and available
            variables.
    """
    present = [v for v in variables if v.code in dataset.data_vars]
    missing = [v for v in variables if v.code not in dataset.data_vars]
    if missing:
        missing_sorted = ", ".join(sorted(v.code for v in missing))
        available = ", ".join(sorted(dataset.data_vars)) or "<none>"
        if not present:
            raise MissingVariableError(
                "Requested variable(s) not present in the decoded GRIB2 file: "
                f"{missing_sorted}. Available variables: {available}."
            )
        logger.warning(
            "Requested variable(s) not present in the decoded GRIB2 file, "
            "recording only the present subset: missing=%s available=%s",
            missing_sorted,
            available,
        )
    return tuple(present)

def _align_store_variables(
    dataset: xr.Dataset,
    present_variables: tuple[VariableSpec, ...],
) -> xr.Dataset:
    """Drop store variables that are absent from the current file.

    When a re-ingested file lacks a variable that earlier files provided
    (e.g. a GEFS ``pgrb2b`` product that omits ``prate``), the lead merge
    would NaN-fill that variable for the new lead while the catalog kept
    advertising it — exactly the catalog-claims-data-that-is-not-there
    state the variable-presence checks are meant to prevent. Dropping the
    variable keeps the store aligned with what the catalog records.

    Args:
        dataset: The merged cycle dataset.
        present_variables: The variables present in the current file.

    Returns:
        The dataset with variables not present in the current file removed.
    """
    present_codes = {variable.code for variable in present_variables}
    extra = [name for name in dataset.data_vars if name not in present_codes]
    if not extra:
        return dataset
    logger.warning(
        "Dropping store variable(s) absent from the current file: %s",
        ", ".join(sorted(str(name) for name in extra)),
    )
    return dataset.drop_vars(extra)

def _normalize_canonical_units(
    dataset: xr.Dataset,
    variables: tuple[VariableSpec, ...],
) -> xr.Dataset:
    """Convert each mapped variable's values to the platform canonical unit.

    The platform stores canonical units in Zarr (Model A, ``docs/API.md``
    section 2.6 / ``docs/MODELS.md`` section 3): GRIB ``K`` temperature becomes
    ``°C`` and GFS ``prate`` (``kg m-2 s-1``) becomes ``mm/h``. The conversion
    is driven by the *actual* GRIB ``units`` attribute on each mapped data
    variable (not by the variable name alone), so a variable that already
    carries the canonical unit is left numerically untouched.

    For each mapped variable present in the dataset:

    * if the ``units`` attribute is absent, the variable is left untouched
      (synthetic in-memory datasets have no GRIB provenance to convert);
    * if the source unit equals the target canonical unit, values are kept and
      the ``units`` attribute is normalized to the canonical string;
    * if the source unit is a known source for the target canonical unit, the
      values are transformed and the ``units`` attribute set to the canonical;
    * otherwise the mapping is rejected with :class:`UnitNormalizationError`
      rather than silently storing values under a possibly-wrong label.

    Args:
        dataset: The mapped dataset (platform variable names).
        variables: The run's :class:`VariableSpec` catalog metadata.

    Returns:
        The dataset with canonical-unit values and ``units`` attributes.

    Raises:
        UnitNormalizationError: If a mapped variable's source unit is present
            but is neither the canonical unit nor a known source for it.
    """
    for variable in variables:
        code = variable.code
        if code not in dataset.data_vars:
            continue
        data_array = dataset[code]
        source_unit = data_array.attrs.get("units")
        if source_unit is None:
            continue
        source_token = _unit_token(str(source_unit))
        target_unit = variable.unit
        if source_token == _unit_token(target_unit):
            data_array.attrs["units"] = target_unit
            continue
        transforms = _SOURCE_TO_CANONICAL.get(target_unit)
        if transforms is None or source_token not in transforms:
            raise UnitNormalizationError(
                f"Cannot canonicalize variable '{code}' from source unit "
                f"'{source_unit}' to canonical unit '{target_unit}': no "
                "supported conversion is defined."
            )
        data_array.values = transforms[source_token](data_array.values)
        data_array.attrs["units"] = target_unit
    return dataset


def ingest_grib_file(
    spec: RunCatalogSpec,
    grib_path: str | Path,
    store_path: str,
    *,
    requested_lead_time_hours: int | None = None,
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
        3. Values are normalized to the platform canonical units (e.g. GRIB
           Kelvin temperature to ``°C``, GFS ``prate`` in ``kg m-2 s-1`` to
           ``mm/h``) and each variable's ``units`` attribute is set to the
           canonical unit (Model A, ``docs/API.md`` section 2.6).
        4. Requested variables are verified against the mapped dataset: a
           completely absent set aborts with :class:`MissingVariableError`;
           a partial subset records only the present variables (missing ones
           are logged).
        5. When ``requested_lead_time_hours`` is given and differs from the
           file's parsed lead, the ingest is aborted with a
           :class:`LeadTimeMismatchError` instead of silently merging or
           overwriting a lead the caller did not ask for (the file's decoded
           lead is authoritative and is never overridden by the request).
        6. The single-lead dataset is merged into the cycle store along
           ``lead_time_hours`` (re-ingesting a lead replaces it).
        7. ``write_dataset`` persists the merged dataset to the store.
        8. ``record_ingested_dataset`` upserts the run (and its catalog rows)
           with ``status='ready'`` so the API serving tier can serve it.

    Args:
        spec: The catalog metadata of the run (model, version, cycle, grid,
            variables).
        grib_path: Path to the downloaded GRIB2 file.
        store_path: Zarr store path/URL of the run's cycle. All leads of a
            cycle share this store.
        requested_lead_time_hours: The lead the caller requested (e.g. the
            CLI's ``--lead-time-hours``). When provided, ingestion fails
            fast if it does not match the file's parsed lead.

    Returns:
        The recorded :class:`ModelRunRecord` in the ``ready`` state.

    Raises:
        GribParsingError: If the GRIB2 file cannot be decoded.
        UnitNormalizationError: If a mapped variable's source unit cannot be
            safely canonicalized.
        MissingVariableError: If a requested variable is absent from the
            parsed file.
        LeadTimeMismatchError: If ``requested_lead_time_hours`` is provided
            and does not match the file's parsed lead.
        ValueError: If the Zarr store target is unsupported.
        CorruptStoreError: If an existing cycle store cannot be opened (it is
            refused rather than silently rebuilt from the current lead).
        sqlalchemy error: If the catalog write fails (no row is committed).
    """
    dataset = parse_grib2(grib_path)
    dataset = _apply_variable_mapping(dataset, spec.variables)
    dataset = _normalize_canonical_units(dataset, spec.variables)
    present_variables = _validate_required_variables(dataset, spec.variables)
    _validate_requested_lead(dataset, requested_lead_time_hours)
    # The read-merge-write cycle on the shared cycle store is serialized by a
    # process-level flock: concurrent workers ingesting different leads of the
    # same run would otherwise race on the shared staging directory and lose
    # updates (review finding MAJOR-1). The corrupt-store check runs inside
    # the lock so it observes a consistent store state.
    with store_lock(store_path):
        status = store_status(store_path)
        if status == "corrupt":
            raise CorruptStoreError(
                f"Existing Zarr store {store_path!r} is corrupt (cannot be "
                "opened); refusing to rebuild it from the current single "
                "lead, which would drop previously ingested leads. Remove "
                "or repair the store before re-ingesting."
            )
        dataset = _merge_lead(dataset, store_path)
        dataset = _align_store_variables(dataset, present_variables)
        write_dataset(dataset, store_path)
    if len(present_variables) != len(spec.variables):
        # Record catalog rows only for the variables actually present in the
        # store (partial-presence degradation).
        spec = RunCatalogSpec(**{**vars(spec), "variables": present_variables})
    return record_ingested_dataset(spec, dataset, effective_store_path=store_path)


def _validate_requested_lead(
    dataset: xr.Dataset,
    requested_lead_time_hours: int | None,
) -> None:
    """Fail fast when a requested lead disagrees with the parsed dataset.

    The GRIB file's decoded lead is the source of truth (``parser.py`` derives
    ``lead_time_hours`` from the ``step`` coordinate), so a mismatch with the
    requested lead aborts ingestion rather than silently re-ingesting an
    unexpected file or overwriting a lead the caller did not ask for.

    Args:
        dataset: The normalized parsed dataset.
        requested_lead_time_hours: The lead the caller requested, or ``None``
            when no request bound is asserted (library callers without a lead
            concept).

    Raises:
        LeadTimeMismatchError: When the requested lead is provided and does
            not match the dataset's parsed lead.
    """
    if requested_lead_time_hours is None:
        return
    lead_values = dataset["lead_time_hours"].values
    parsed = int(lead_values[0] if np.ndim(lead_values) != 0 else lead_values)
    if parsed != requested_lead_time_hours:
        raise LeadTimeMismatchError(
            f"Downloaded GRIB2 file decodes to lead time {parsed}h, but the "
            f"requested lead time is {requested_lead_time_hours}h. The file "
            "does not match the requested forecast; aborting instead of "
            "ingesting an unexpected lead."
        )


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