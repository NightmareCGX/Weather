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

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr

from ingestion.core.base import (
    CycleStoreMismatchError,
    IngestionError,
    LeadTimeMismatchError,
    StoreSchemaMismatchError,
)
from ingestion.core.catalog import (
    ModelRunRecord,
    RunCatalogSpec,
    VariableSpec,
    record_ingested_dataset,
)
from ingestion.core.zarr_writer import (
    commit_region,
    prepare_run_store,
    read_dataset,
    store_exists,
)
from ingestion.providers.noaa.parser import parse_grib2


class UnitNormalizationError(IngestionError):
    """Raised when a source GRIB unit cannot be safely canonicalized.

    A mapped variable whose source ``units`` attribute is neither already equal
    to the target canonical unit nor a known source unit for that target is
    rejected rather than silently stored under a possibly-wrong label.
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
    member: int | None = None,
) -> ModelRunRecord:
    """Parse a GRIB2 file, write it to a Zarr store, and record it in the catalog.

    This is the production orchestration call path for an already-downloaded
    forecast file. ``store_path`` is the store of the run's *cycle*: one
    ``model_runs`` row represents a full forecast cycle and its store
    accumulates every lead/member (DATABASE.md ``UNIQUE(model_version_id,
    cycle_time)``).

    Write path: the serving store is pre-allocated (a full store covering the
    run's expected leads and, for GEFS, members) and each file is committed to
    its own region:

        1. ``parse_grib2`` decodes and normalizes the GRIB2 file.
        2. Data variables are renamed to the platform vocabulary (per
           ``spec.variables``).
        3. Values are normalized to the platform canonical units.
        4. When ``requested_lead_time_hours`` is given and differs from the
           file's parsed lead, the ingest is aborted with a
           :class:`LeadTimeMismatchError` (the file's decoded lead is
           authoritative).
        5. The single-lead (and, for GEFS, single-member) dataset is written
           to the store via a targeted ``region`` write at the coordinate-
           driven (lead, member) position. Re-ingesting a file replaces only
           its own region; other leads/members are never read or rewritten.
        6. ``record_ingested_dataset`` records the file's product/member rows
           without marking the whole run ready (run-level readiness is decided
           separately by completeness).

    Args:
        spec: The catalog metadata of the run (model, version, cycle, grid,
            variables).
        grib_path: Path to the downloaded GRIB2 file.
        store_path: Zarr store path/URL of the run's cycle.
        requested_lead_time_hours: The lead the caller requested (e.g. the
            CLI's ``--lead-time-hours``). When provided, ingestion fails
            fast if it does not match the file's parsed lead.
        member: The upstream GEFS member identity (``1..30``) for a
            per-member file. ``None`` for deterministic models or combined
            files. The member identity maps to the store's ``member``
            coordinate position, so ``gep17`` lands in member 17 regardless of
            completion order.

    Returns:
        The recorded :class:`ModelRunRecord` (in its current, possibly
        non-ready, state).

    Raises:
        GribParsingError: If the GRIB2 file cannot be decoded.
        UnitNormalizationError: If a mapped variable's source unit cannot be
            safely canonicalized.
        LeadTimeMismatchError: If ``requested_lead_time_hours`` is provided
            and does not match the file's parsed lead.
        ValueError: If the Zarr store target is unsupported.
        sqlalchemy error: If the catalog write fails (no row is committed).
    """
    dataset = parse_grib2(grib_path)
    dataset = _apply_variable_mapping(dataset, spec.variables)
    dataset = _normalize_canonical_units(dataset, spec.variables)
    # Record the model so the Zarr store is self-describing about its forecast
    # run (model + cycle) independently of the S3 path (ACCEPTANCE_REMEDIATION
    # PLAN §4). ``cycle_time`` is set by the parser from the GRIB ``time``
    # coordinate.
    dataset.attrs["model_id"] = spec.model_id
    _validate_requested_lead(dataset, requested_lead_time_hours)
    _commit_region(
        dataset,
        store_path,
        member=member,
        expected_lead_time_hours=spec.expected_lead_time_hours,
        expected_members=spec.expected_members,
    )
    return record_ingested_dataset(
        spec, dataset, effective_store_path=store_path, member=member
    )


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

    A Zarr store represents exactly one forecast cycle
    (``UNIQUE(model_version_id, cycle_time)`` per DATABASE.md), so **before**
    merging the incoming dataset's cycle identity is validated against the
    existing store's identity. A mismatch is a hard error
    (:class:`CycleStoreMismatchError`) — the merge is refused, never silently
    bypassed. Same-cycle leads are also validated for structural compatibility
    (grid axes, member axis, per-variable dimensions).

    Args:
        dataset: The normalized single-lead dataset for this GRIB file.
        store_path: The cycle store to merge into (may not exist yet).

    Returns:
        The dataset for the whole cycle: the new lead alone on a fresh ingest,
        otherwise the accumulated cycle with the new lead merged in.

    Raises:
        CycleStoreMismatchError: If the incoming dataset's cycle differs from
            the existing store's cycle, or either identity cannot be
            established.
        StoreSchemaMismatchError: If the incoming lead is structurally
            incompatible with the existing store.
    """
    if "lead_time_hours" not in dataset.dims:
        dataset = dataset.expand_dims("lead_time_hours")
    if not store_exists(store_path):
        return dataset

    existing = read_dataset(store_path)
    _validate_store_identity(dataset, existing, store_path)
    _validate_lead_schema(dataset, existing, store_path)
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


def _commit_region(
    dataset: xr.Dataset,
    store_path: str,
    *,
    member: int | None = None,
    expected_lead_time_hours: tuple[int, ...] = (),
    expected_members: tuple[int, ...] = (),
) -> None:
    """Commit a single-lead (and optional single-member) file to the store.

    The file's ``lead_time_hours`` (and, for a GEFS per-member file, the
    ``member`` coordinate value) determines the exact region of the
    pre-allocated serving store that is written. Only that region is touched:
    existing data in other leads/members is never read or rewritten.

    When the store does not yet exist, it is **pre-allocated** with the full
    expected coordinate structure (``expected_lead_time_hours`` and, for GEFS,
    ``expected_members``) so every subsequent file region-commits independently
    — no whole-store read-modify-write ever happens.

    Args:
        dataset: The normalized single-lead dataset for this GRIB file.
        store_path: The cycle store to commit into (may not exist yet).
        member: The upstream GEFS member identity (``1..30``). ``None`` for
            deterministic models. When set, the dataset's ``member``
            coordinate is replaced with ``[member]`` so the region mapping is
            coordinate-driven.
        expected_lead_time_hours: The full lead set the run is expected to
            serve; used to pre-allocate the store on the first write.
        expected_members: The full GEFS member set; used to pre-allocate the
            store on the first write.

    Raises:
        CycleStoreMismatchError: If the store already exists and the incoming
            cycle differs from the store's cycle.
        StoreSchemaMismatchError: If the incoming lead/member is structurally
            incompatible with the store.
    """
    # Ensure the lead is a dimension (length 1) so region writes map it
    # positionally; the parser emits it as a scalar coordinate.
    if "lead_time_hours" not in dataset.dims:
        dataset = dataset.expand_dims("lead_time_hours")

    if member is not None:
        if "member" not in dataset.dims and "member" not in dataset.coords:
            raise StoreSchemaMismatchError(
                "A GEFS member identity was supplied but the parsed dataset "
                "has no member coordinate or dimension."
            )
        # Pin the member coordinate to the real upstream identity so the
        # region mapping below is coordinate-driven (gep17 -> member 17).
        dataset = dataset.assign_coords(member=[int(member)])
        # Promote a scalar member coordinate to a length-1 dimension AND
        # broadcast each data variable onto it, so the region write maps the
        # member positionally and the data var dims match the store's
        # ``(member, ...)`` layout.
        if "member" not in dataset.dims:
            dataset = dataset.expand_dims("member")
        # If the data vars still lack the member dim (a scalar-member file
        # decodes t2m as ``(latitude, longitude)``), broadcast them onto it.
        need_member = [
            name
            for name in dataset.data_vars
            if "member" not in dataset[name].dims
        ]
        if need_member:
            dataset = dataset.assign(
                {
                    name: dataset[name].expand_dims("member")
                    for name in need_member
                }
            )

    if not store_exists(store_path):
        # First write: pre-allocate the full serving store with the run's
        # expected leads (and, for GEFS, members), NaN-filled, then commit this
        # file's own region so the store immediately carries real data.
        prepare_run_store(
            dataset,
            store_path,
            expected_lead_time_hours=expected_lead_time_hours,
            expected_members=expected_members,
        )
        commit_region(dataset, store_path)
        return

    existing = read_dataset(store_path)
    _validate_store_identity(dataset, existing, store_path)
    _validate_lead_schema(dataset, existing, store_path)

    commit_region(dataset, store_path)


def _resolve_cycle_time(dataset: xr.Dataset) -> str | None:
    """Return a dataset's forecast-run cycle/reference time as a UTC string.

    The authoritative identity is the ``cycle_time`` attribute written by the
    parser from the GRIB ``time`` coordinate. Legacy stores written before that
    attribute existed still carry the ``time`` coordinate (the parser always
    kept it), which is used as the fallback so old stores remain validable.

    Args:
        dataset: A normalized dataset (incoming lead or existing store).

    Returns:
        The cycle time as an ISO 8601 UTC string, or ``None`` when the dataset
        carries no cycle identity.
    """
    if "cycle_time" in dataset.attrs:
        return str(dataset.attrs["cycle_time"])
    if "time" in dataset.coords:
        value = dataset.coords["time"].values
        item = value.item() if np.ndim(value) != 0 else value
        # ``np.datetime_as_string`` on a scalar returns a 0-d ndarray;
        # ``item()`` extracts the plain ``str`` so the type is ``str | None``.
        return str(np.datetime_as_string(np.asarray(item, dtype="datetime64[ns]"), unit="s").item())
    return None


def _validate_store_identity(
    dataset: xr.Dataset,
    existing: xr.Dataset,
    store_path: str,
) -> None:
    """Refuse to merge a dataset whose cycle differs from the store's cycle.

    Fail-fast correctness: a Zarr store represents one forecast cycle and must
    never silently accept data belonging to another cycle. The merge is gated
    on the incoming and stored identities matching exactly; if either identity
    cannot be established the merge is refused (rather than guessing).

    Args:
        dataset: The incoming single-lead dataset.
        existing: The existing cycle store dataset.
        store_path: The store path (used in the error message).

    Raises:
        CycleStoreMismatchError: If the cycles differ or an identity is missing.
    """
    requested = _resolve_cycle_time(dataset)
    stored = _resolve_cycle_time(existing)
    if requested is None:
        raise CycleStoreMismatchError(
            f"Refusing to merge into {store_path!r}: the incoming forecast has "
            "no cycle/reference time, so its forecast-run identity cannot be "
            "established."
        )
    if stored is None:
        raise CycleStoreMismatchError(
            f"Refusing to merge into {store_path!r}: the existing store carries "
            "no cycle/reference time, so it cannot be identified as a valid "
            "cycle store."
        )
    if requested != stored:
        raise CycleStoreMismatchError(
            f"Refusing to merge: the incoming forecast is cycle {requested}, "
            f"but the store at {store_path!r} already contains cycle {stored}. "
            "A Zarr store represents exactly one forecast cycle; the cycles "
            "must match."
        )


def _validate_lead_schema(
    dataset: xr.Dataset,
    existing: xr.Dataset,
    store_path: str,
) -> None:
    """Validate that a same-cycle lead/member is structurally compatible.

    Same-cycle files committed into a cycle store must share the same spatial
    grid (latitude/longitude axes); a variable present in both must have the
    same dimensions (except that a GEFS per-member file's ``member`` dimension
    of length 1 is compatible with a multi-member store — the region write
    targets exactly one member). Adding a previously-absent variable is
    allowed (a lead file may omit a field the parser skips); redefining an
    existing variable's structure is not.

    Args:
        dataset: The incoming single-lead (optionally single-member) dataset.
        existing: The existing cycle store dataset.
        store_path: The store path (used in the error message).

    Raises:
        StoreSchemaMismatchError: If the incoming file is incompatible.
    """
    for axis in ("latitude", "longitude"):
        if axis in existing.coords and axis in dataset.coords:
            if not np.array_equal(existing.coords[axis].values, dataset.coords[axis].values):
                raise StoreSchemaMismatchError(
                    f"Refusing to merge into {store_path!r}: the incoming file's "
                    f"'{axis}' axis differs from the store's '{axis}' axis. A "
                    "cycle store must have one consistent grid."
                )
    for code in set(dataset.data_vars) & set(existing.data_vars):
        # A per-member GEFS file has a length-1 member dim; the store has the
        # full member axis. Allow the member dim to differ (region write
        # targets one member), but all other dims must match exactly.
        incoming_dims = set(dataset[code].dims)
        existing_dims = set(existing[code].dims)
        if "member" in incoming_dims and "member" not in existing_dims:
            raise StoreSchemaMismatchError(
                f"Refusing to merge into {store_path!r}: variable '{code}' has "
                f"a member dimension in the incoming file but the store does not."
            )
        if "member" in existing_dims and "member" in incoming_dims:
            # The non-member dims must match; the member axis length differs by
            # design (single-member file into multi-member store).
            incoming_dims.discard("member")
            existing_dims.discard("member")
        if incoming_dims != existing_dims:
            raise StoreSchemaMismatchError(
                f"Refusing to merge into {store_path!r}: variable '{code}' has "
                f"dimensions {tuple(dataset[code].dims)} in the incoming file "
                f"but {tuple(existing[code].dims)} in the store."
            )
