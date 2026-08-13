"""GRIB2 parsing and normalization for the NOAA NOMADS provider.

Raw GRIB2 files are decoded with ``cfgrib`` (backed by ``eccodes``) into
an ``xarray.Dataset`` and normalized to the platform convention described
in ``docs/DATABASE.md``:

* ``lead_time_hours`` is derived from the GRIB ``step`` offset instead of
  an absolute ``valid_time``, keeping products idempotent across cycles.
* ``time``, ``latitude``, ``longitude`` (and the level coordinate) are
  retained as the primary dimensions and coordinates.
* The GEFS ensemble dimension is normalized from cfgrib's ``number`` name to
  the platform's ``member`` convention, so the rest of the pipeline and the
  API uniformly expose "ensemble member".

Operational NOAA GFS/GEFS ``pgrb2`` products are large multi-message files
spanning many vertical levels and ``typeOfLevel`` values. cfgrib treats
``typeOfLevel`` (and several other keys) as *unique* per dataset, so opening
the whole file as one unfiltered ``xr.Dataset`` raises
``DatasetBuildError: multiple values for unique key 'typeOfLevel'``. The
parser therefore selects each platform-required surface field with an
explicit ``filter_by_keys`` selection, normalizes each selected field, and
merges the results into a single dataset.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from ingestion.core.base import IngestionError


class GribParsingError(IngestionError):
    """Raised when a GRIB2 file cannot be decoded or normalized."""


#: cfgrib ``filter_by_keys`` selections for each platform-required surface
#: field. Each entry maps the cfgrib-emitted data-variable name (the GRIB
#: ``cfVarName``) to the exact GRIB metadata that uniquely selects that field
#: from an operational multi-message GFS/GEFS ``pgrb2`` file. The selectors are
#: deliberately narrow: temperature is pinned to ``heightAboveGround`` level 2
#: and precipitation rate to ``surface`` level 0 with ``stepType=instant`` (a
#: ``pgrb2`` file carries both instant and time-averaged ``prate`` messages).
#: WARNING — every new platform surface variable MUST be registered here in
#: addition to the catalog/CLI ``VariableSpec`` metadata. ``parse_grib2`` only
#: decodes the fields listed below, so an unregistered variable (e.g. a 10 m
#: wind selected via a custom ``--variable``) would be absent from the
#: dataset and correctly rejected by the pipeline's variable-missing guard.

SURFACE_FIELD_FILTERS: dict[str, dict[str, object]] = {
    # GRIB shortName ``2t`` = "2 metre temperature" (paramId 167), K.
    "t2m": {
        "shortName": "2t",
        "typeOfLevel": "heightAboveGround",
        "level": 2,
        "stepType": "instant",
    },
    # GRIB shortName ``prate`` = "Precipitation rate" (paramId 7), kg m-2 s-1.
    "prate": {
        "shortName": "prate",
        "typeOfLevel": "surface",
        "level": 0,
        "stepType": "instant",
    },
}


def parse_grib2(path: str | Path) -> xr.Dataset:
    """Decode a single GRIB2 file into a normalized, in-memory Dataset.

    Each platform-required surface field is selected from the file with an
    explicit :data:`SURFACE_FIELD_FILTERS` selection, normalized, and merged
    into the returned dataset. Fields not present in the product (e.g. a GEFS
    ``pgrb2b`` file that omits a variable) are skipped rather than failing the
    whole parse. The GEFS ensemble ``number`` dimension is preserved through
    the normalization step and becomes the platform ``member`` dimension.

    The returned dataset is fully loaded into memory and its file handle
    closed before the method returns. Coordinates are normalized so the
    dataset is immediately writable to a Zarr store:

    * ``step`` / ``valid_time`` are dropped.
    * ``lead_time_hours`` is added as the integer forecast offset
      (hours) derived from the GRIB ``step`` timedelta.

    Args:
        path: Path to a GRIB2 file.

    Returns:
        A normalized ``xarray.Dataset``.

    Raises:
        GribParsingError: If the file cannot be opened, decoded, the
            ``step`` coordinate is missing, or none of the platform-required
            surface fields can be selected.
    """
    selected: list[xr.Dataset] = []
    for variable_name, filter_by_keys in SURFACE_FIELD_FILTERS.items():
        try:
            with xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": filter_by_keys},
            ) as raw:
                dataset = raw.load()
        except Exception as exc:  # cfgrib/eccodes raise varied exceptions
            raise GribParsingError(
                f"Failed to decode GRIB2 file {path!s}: {exc}"
            ) from exc

        if not dataset.data_vars:
            # The field is not present in this product; skip it rather than
            # failing the whole parse (a zero-match filter returns an empty
            # dataset, not an error).
            continue

        selected.append(normalize(dataset))

    if not selected:
        raise GribParsingError(
            f"Decoded GRIB2 file {path!s} contains none of the required "
            "surface fields: " + ", ".join(SURFACE_FIELD_FILTERS) + "."
        )

    if len(selected) == 1:
        return selected[0]
    return xr.merge(selected)


def normalize(dataset: xr.Dataset) -> xr.Dataset:
    """Normalize a decoded GRIB2 dataset to the platform convention.

    Args:
        dataset: Dataset as decoded by ``cfgrib`` (loaded into memory).

    Returns:
        The dataset with ``lead_time_hours`` replacing the ``step``
        coordinate.

    Raises:
        GribParsingError: If the ``step`` coordinate is absent.
    """
    if "step" not in dataset.coords:
        raise GribParsingError(
            "Decoded GRIB2 dataset has no 'step' coordinate; "
            "cannot derive lead_time_hours."
        )

    step = dataset.coords["step"]
    if step.size != 1:
        raise GribParsingError(
            "Expected a single forecast step per GRIB2 file, "
            f"got {step.size} steps."
        )

    step_value = step.values.item()
    lead_time_hours = int(np.timedelta64(step_value, "ns") / np.timedelta64(1, "h"))

    # cfgrib decodes the GRIB ``number`` key (the GEFS perturbation number) as
    # a dataset dimension/coordinate named ``number``. The platform convention
    # is ``member`` (the API and catalog key on it), so normalize the name here
    # and keep the rest of the pipeline uniformly on ``member``.
    if "number" in dataset.dims or "number" in dataset.coords:
        dataset = dataset.rename({"number": "member"})

    normalized = dataset.drop_vars(["step", "valid_time"], errors="ignore")
    normalized = normalized.assign_coords(lead_time_hours=lead_time_hours)
    return normalized