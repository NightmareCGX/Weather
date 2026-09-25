"""Authoritative storage-dtype resolver for the sharded store formats.

Single source of truth for "which dtype does a variable's shard payload carry"
per storage format version and product role. The writer (``zarr_writer``), the
API readers (``ShardedV1Reader``/``ShardedV2Reader``), the ingestion read-back
helpers (``read_slice``/``_populate_sharded_data``), inventory, and fixtures all
derive their dtypes from here — no file may keep its own copy of the dtype
matrix.

Frozen contract (``docs/investigations/float16-storage-feasibility/``
``QUANTIZATION_BENCHMARK.md`` + ``SHARDED_V2_IMPLEMENTATION_PLAN.md`` §4):

* ``sharded_v1``: every variable stored as float32 (historical bytes, frozen).
* ``sharded_v2``:
  - nine continuous fields as little-endian float16 (``<f2``);
  - ``precipitation_amount_3h`` and ``cloud_ceiling`` as little-endian float32
    (``<f4``) — *semantic compatibility exceptions*, not precision limits:
    precipitation guards the exactly-``0.10 mm`` dry/wet threshold (GEFS decimal
    packing produces a large mass of values exactly equal to ``f32(0.1)``), and
    cloud ceiling guards the ``19.99 km`` unlimited sentinel (real float16
    classification flips observed in a ~2.19 m window above the threshold);
  - deterministic/member categorical precipitation flags as ``u1``;
  - ensemble-mean flags as little-endian float32 — they are member-mean
    probabilities in ``[0, 1]``, not binary categoricals.

Threshold-coupled variable rule: if product logic compares a persisted value
directly against a threshold/sentinel, a new variable must NOT silently enter
float16. Onboarding requires the threshold, comparison domain, source packing,
float16 grid, and real-data distribution around the threshold to be reviewed;
without a demonstrated isolation band the variable stays float32. The matrix
below is authoritative — do not widen ``_V2_FLOAT16_VARIABLES`` mechanically.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from domain.reclamation import TARGET_KIND_DET, TARGET_KIND_MEAN, TARGET_KIND_MEM

#: Legacy unsharded fallback version string (manifest absent). Stored values are
#: read through the xarray path, not shard payloads; named confusingly close to
#: ``sharded_v2`` but entirely unrelated to it.
LEGACY_UNSHARDED_FORMAT_VERSION = "v2_unsharded"

#: Frozen historical format. All shard payloads are float32; byte behavior is
#: frozen and must never change.
SHARDED_V1_FORMAT_VERSION = "sharded_v1"

#: Sharded v2: per-variable native dtypes (this module's matrix).
SHARDED_V2_FORMAT_VERSION = "sharded_v2"

#: Format versions whose shard payloads are decoded through the sharded-reader
#: path (as opposed to the legacy unsharded xarray path).
_SHARDED_PAYLOAD_FORMATS: frozenset[str] = frozenset(
    {SHARDED_V1_FORMAT_VERSION, SHARDED_V2_FORMAT_VERSION}
)

#: sharded_v2 continuous fields stored as little-endian float16.
_V2_FLOAT16_VARIABLES: frozenset[str] = frozenset(
    {
        "temperature_2m",
        "relative_humidity_2m",
        "wind_u_10m",
        "wind_v_10m",
        "wind_gust",
        "precipitation_rate",
        "cloud_cover_3h",
        "snow_depth",
        "visibility",
    }
)

#: sharded_v2 float32 semantic-compatibility exceptions (threshold-coupled).
_V2_FLOAT32_VARIABLES: frozenset[str] = frozenset(
    {"precipitation_amount_3h", "cloud_ceiling"}
)

#: Categorical precipitation flags (Code table 4.222 indicators).
_FLAG_VARIABLES: frozenset[str] = frozenset({"crain", "csnow", "cfrzr", "cicep"})

#: Every variable the sharded_v2 dtype matrix knows about. New variables must be
#: added here explicitly (fail-closed for anything else).
SHARDED_V2_KNOWN_VARIABLES: frozenset[str] = (
    _V2_FLOAT16_VARIABLES | _V2_FLOAT32_VARIABLES | _FLAG_VARIABLES
)

#: Explicit little-endian payload dtypes. ``np.dtype("<f2")`` keeps its
#: little-endian layout on any host, so ``ndarray.tobytes()`` of an array cast
#: through these dtypes is a stable cross-platform byte contract.
FLOAT16_LE_DTYPE = np.dtype("<f2")
FLOAT32_LE_DTYPE = np.dtype("<f4")
UINT8_DTYPE = np.dtype("u1")

#: The frozen sharded_v1 payload dtype: float32 for every variable and role.
SHARDED_V1_PAYLOAD_DTYPE = np.dtype("<f4")


def is_sharded_payload_format(format_version: str) -> bool:
    """Return True for format versions whose payloads are sharded containers.

    ``sharded_v1`` and ``sharded_v2`` both read through the sharded-reader
    path. ``v2_unsharded`` (the legacy unsharded fallback) does not.
    """
    return format_version in _SHARDED_PAYLOAD_FORMATS


def resolve_storage_dtype(
    format_version: str,
    variable: str,
    product_role: str,
) -> np.dtype[Any]:
    """Resolve the shard-payload dtype for one variable under one format.

    Args:
        format_version: ``sharded_v1`` or ``sharded_v2``. Anything else fails
            closed (an unknown format must never get a guessed dtype).
        variable: Canonical platform variable code (e.g. ``temperature_2m``).
        product_role: ``det`` / ``mem`` / ``mean`` (the ``TARGET_KIND_*``
            vocabulary from :mod:`domain.reclamation`). The role participates in
            the decision because the categorical precipitation flags are binary
            indicators for deterministic/member shards but member-mean
            probabilities for the ensemble-mean product.

    Returns:
        The explicit little-endian payload dtype.

    Raises:
        ValueError: If the format version is unknown, the product role is
            unknown, or the variable has no explicit sharded_v2 dtype decision
            (new variables must be onboarded through review, never defaulted).
    """
    if format_version == SHARDED_V1_FORMAT_VERSION:
        return SHARDED_V1_PAYLOAD_DTYPE
    if format_version != SHARDED_V2_FORMAT_VERSION:
        raise ValueError(
            f"Unknown storage format version {format_version!r}: no dtype "
            "decision exists. Expected "
            f"{SHARDED_V1_FORMAT_VERSION!r} or {SHARDED_V2_FORMAT_VERSION!r}."
        )

    if product_role not in (TARGET_KIND_DET, TARGET_KIND_MEM, TARGET_KIND_MEAN):
        raise ValueError(
            f"Unknown product role {product_role!r} for variable {variable!r}: "
            f"expected one of {TARGET_KIND_DET!r}, {TARGET_KIND_MEM!r}, "
            f"{TARGET_KIND_MEAN!r}."
        )

    if variable in _V2_FLOAT16_VARIABLES:
        return FLOAT16_LE_DTYPE
    if variable in _V2_FLOAT32_VARIABLES:
        return FLOAT32_LE_DTYPE
    if variable in _FLAG_VARIABLES:
        # Binary indicators for det/member shards; member-mean probabilities
        # (float32) for the ensemble-mean product. Same variable name, different
        # semantics — the role must always participate in the decision.
        if product_role == TARGET_KIND_MEAN:
            return FLOAT32_LE_DTYPE
        return UINT8_DTYPE

    raise ValueError(
        f"Variable {variable!r} has no sharded_v2 dtype decision: a new "
        "variable requires an explicit sharded_v2 dtype review (threshold "
        "coupling, source packing, float16 grid, and real-data distribution "
        "around any product threshold) before it can be persisted. Refusing "
        "to default the dtype."
    )
