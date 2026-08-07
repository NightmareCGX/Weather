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
"""

from pathlib import Path

import numpy as np
import xarray as xr

from ingestion.core.base import IngestionError


class GribParsingError(IngestionError):
    """Raised when a GRIB2 file cannot be decoded or normalized."""


def parse_grib2(path: str | Path) -> xr.Dataset:
    """Decode a single GRIB2 file into a normalized, in-memory Dataset.

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
        GribParsingError: If the file cannot be opened, decoded, or the
            ``step`` coordinate is missing.
    """
    try:
        with xr.open_dataset(path, engine="cfgrib") as raw:
            dataset = raw.load()
    except Exception as exc:  # cfgrib/eccodes raise varied exceptions
        raise GribParsingError(f"Failed to decode GRIB2 file {path!s}: {exc}") from exc

    return normalize(dataset)


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
