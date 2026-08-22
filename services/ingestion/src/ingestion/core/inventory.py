"""Physical object-inventory for the region-write completion evidence.

A COMPLETE marker attests that a compliant writer completed a logical region.
The marker body records:

* ``expected_write_set_fingerprint`` — sha256 of the expected write set;
* ``required_materialized_object_keys`` — physical object keys that MUST exist;
* ``intentionally_omitted_fill_chunks`` — physical chunk coordinates that are
  all-fill (write_empty_chunks=False) and deliberately absent.

The coalesced finalizer validates this evidence:

* materialized and omitted sets are disjoint and their union covers the
  expected write set;
* every required materialized object currently exists;
* intentionally omitted fill chunks may be physically absent;
* under the ensemble shared-member-chunk layout, object existence does NOT
  prove member-slice completion — the COMPLETE marker does.

External deletion of a required materialized object invalidates committed
state (the region becomes uncommitted), preserving external-shrink detection.
"""

from __future__ import annotations

import json
import os

import s3fs  # type: ignore[import-untyped]

from domain.locks import sha256_hex


class InventoryError(RuntimeError):
    """Raised when marker completion evidence is structurally invalid."""


def _storage_backend(store_path: str) -> tuple[str, str]:
    """Resolve a store path to (backend_kind, backend_root)."""
    if store_path.startswith("s3://"):
        rest = store_path[len("s3://") :].strip("/")
        return "s3", rest
    path = store_path
    if path.startswith("file://"):
        path = path[len("file://") :]
    return "local", os.path.abspath(os.path.normpath(path))


def _s3_fs() -> "s3fs.S3FileSystem":
    from ingestion.core.config import settings

    scheme = "https" if settings.MINIO_SECURE else "http"
    return s3fs.S3FileSystem(
        key=settings.MINIO_ACCESS_KEY,
        secret=settings.MINIO_SECRET_KEY,
        client_kwargs={"endpoint_url": f"{scheme}://{settings.MINIO_ENDPOINT}"},
    )


def object_exists(store_path: str, object_key: str) -> bool:
    """Return whether a physical object key exists in the store."""
    backend, root = _storage_backend(store_path)
    if backend == "local":
        full = os.path.join(root, *object_key.split("/"))
        return os.path.isfile(full)
    fs = _s3_fs()
    full = f"{root}/{object_key}"
    try:
        result = fs.exists(full)
        return bool(result)
    except Exception:  # noqa: BLE001 - a transient stat failure is a miss
        return False


def list_object_keys(store_path: str, array_path: str) -> list[str]:
    """Return the physical chunk object keys under a data-variable path.

    Args:
        store_path: The store path/URL.
        array_path: The Zarr array path (e.g. ``"temperature_2m"``).

    Returns:
        A sorted list of relative object keys (e.g.
        ``"temperature_2m/0.0.0"``).
    """
    backend, root = _storage_backend(store_path)
    if backend == "local":
        base = os.path.join(root, *array_path.split("/"))
        out: list[str] = []
        if os.path.isdir(base):
            for name in os.listdir(base):
                if name.startswith("."):
                    continue
                out.append(f"{array_path}/{name}")
        return sorted(out)
    fs = _s3_fs()
    full_prefix = f"{root}/{array_path}"
    out = []
    try:
        for item in fs.find(full_prefix):
            rel = item[len(root) + 1 :]
            if "/" in rel and not rel.split("/")[-1].startswith("."):
                out.append(rel)
    except Exception:  # noqa: BLE001 - unreadable prefix -> empty
        pass
    return sorted(out)


def expected_write_set_fingerprint(required_keys: list[str], omitted: list[str]) -> str:
    """Fingerprint the expected write set from materialized + omitted keys."""
    return sha256_hex("write-set", *sorted(required_keys), *sorted(omitted))


def _member_positional_index(store_path: str, array_path: str, member: int) -> int:
    """Return the positional index of an ensemble member in the store's member
    coordinate (the upstream member identity may be 1..30, not positional).

    Falls back to ``member - 1`` when the member coordinate cannot be read.
    """
    try:
        from ingestion.core.zarr_writer import read_dataset

        ds = read_dataset(store_path)
        if "member" in ds.coords:
            values = ds.coords["member"].values
            flat = list(values)
            for i, value in enumerate(flat):
                if int(value) == int(member):
                    return i
    except Exception:  # noqa: BLE001 - fall back to identity-1
        pass
    return max(0, member - 1)


def physical_conflict_keys(
    store_path: str,
    *,
    member: int | None,
    lead_index: int,
    data_var_paths: list[str],
) -> list[str]:
    """Derive the physical-chunk conflict identities a logical region writes.

    This maps a logical region ``(member?, lead, lat, lon)`` to the actual
    physical chunk coordinates it can modify, using the store's real
    ``.zarray`` chunk geometry (NOT a hard-coded member/lead convention).

    Under the CURRENT ensemble layout (``member`` full-extent, ``lead``
    chunked at 1), different members at the same lead map to the SAME member
    chunk coordinate (0) and the same lead chunk -> identical conflict keys,
    so they serialize. Different leads map to different lead chunks -> disjoint
    keys.

    For a layout where ``member`` is chunked at 1 (a test-only alternate
    layout), different members map to different member-chunk coordinates ->
    distinct keys.

    Args:
        store_path: The store path/URL.
        member: The ensemble member identity, or ``None`` for deterministic.
        lead_index: The positional index of the lead in the store's
            ``lead_time_hours`` coordinate.
        data_var_paths: The data-variable array paths.

    Returns:
        A sorted, de-duplicated list of physical-chunk conflict identities
        (e.g. ``"temperature_2m/0.0.0"`` for a full-member ensemble chunk).
    """
    out: list[str] = []
    for array_path in data_var_paths:
        za = _read_zarray(store_path, array_path)
        if za is None:
            continue
        shape = za.get("shape")
        chunks = za.get("chunks")
        if not isinstance(shape, list) or not isinstance(chunks, list):
            continue
        shape = [int(s) for s in shape]
        chunks = [int(c) for c in chunks]
        # Layout: (member?, lead, lat, lon). Lead is dim index 1 for ensemble
        # (4-D), index 0 for deterministic (3-D).
        lead_dim = 1 if len(shape) == 4 else 0
        has_member = len(shape) == 4
        # The member axis (if present) is at index 0. Compute the member chunk
        # coordinate from the ACTUAL chunk size using the member's POSITIONAL
        # index in the store's member coordinate (the member identity may be a
        # 1..30 upstream number, not a positional index). For a full-extent
        # member chunk, every member positional index maps to member chunk 0.
        member_chunk = 0
        if has_member and member is not None:
            member_index = _member_positional_index(store_path, array_path, member)
            member_chunk = member_index // chunks[0] if chunks[0] > 0 else 0
        n_spatial = len(shape) - lead_dim - 1
        spatial_chunk_counts = [
            (shape[lead_dim + 1 + i] + chunks[lead_dim + 1 + i] - 1)
            // chunks[lead_dim + 1 + i]
            for i in range(n_spatial)
        ]
        import itertools

        for combo in itertools.product(*[range(c) for c in spatial_chunk_counts]):
            coords = [0] * len(shape)
            if has_member:
                coords[0] = member_chunk
                coords[lead_dim] = lead_index
                coords[lead_dim + 1 :] = list(combo)
            else:
                coords[lead_dim] = lead_index
                coords[lead_dim + 1 :] = list(combo)
            key = array_path + "/" + ".".join(str(c) for c in coords)
            out.append(key)
    return sorted(set(out))


def _read_zarray(store_path: str, array_path: str) -> dict[str, object] | None:
    """Read a data variable's ``.zarray`` metadata, or None when absent."""
    backend, root = _storage_backend(store_path)
    rel = f"{array_path}/.zarray"
    if backend == "local":
        full = os.path.join(root, *rel.split("/"))
        try:
            with open(full, "r", encoding="utf-8") as fh:
                parsed = json.load(fh)
                return parsed if isinstance(parsed, dict) else None
        except (OSError, json.JSONDecodeError):
            return None
    fs = _s3_fs()
    full = f"{root}/{rel}"
    try:
        raw = fs.cat_file(full)
        parsed = json.loads(raw.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else None
    except (Exception, json.JSONDecodeError):  # noqa: BLE001 - unreadable -> None
        return None


def region_expected_object_keys(
    store_path: str,
    *,
    member: int | None,
    lead_index: int,
    data_var_paths: list[str],
    zarray_cache: dict[str, dict[str, object]] | None = None,
) -> list[str]:
    """Derive the physical chunk object keys a logical region writes.

    The actual store's ``.zarray`` chunk metadata determines the physical chunk
    grid. Under the current layout (``lead`` chunked at 1, ``member``
    full-extent for ensemble), a logical region maps to the spatial chunks of
    the target lead (all spatial chunks across each data variable).

    Args:
        store_path: The store path/URL.
        member: The ensemble member identity, or ``None`` for deterministic.
        lead_index: The positional index of the lead in the store's
            ``lead_time_hours`` coordinate.
        data_var_paths: The data-variable array paths (e.g.
            ``["temperature_2m"]``).
        zarray_cache: An optional in-memory cache of per-array ``.zarray``
            metadata (built once per finalizer run), avoiding repeated remote
            reads across markers.

    Returns:
        A sorted list of physical chunk object keys (relative to the store).
    """
    out: list[str] = []
    for array_path in data_var_paths:
        za: dict[str, object] | None
        if zarray_cache is not None and array_path in zarray_cache:
            za = zarray_cache[array_path]
        else:
            za = _read_zarray(store_path, array_path)
            if za is not None and zarray_cache is not None:
                zarray_cache[array_path] = za
        if za is None:
            continue
        shape = za.get("shape")
        chunks = za.get("chunks")
        if not isinstance(shape, list) or not isinstance(chunks, list):
            continue
        shape = [int(s) for s in shape]
        chunks = [int(c) for c in chunks]
        # Determine the position of the lead dim. The layout is
        # (member?, lead, lat, lon). Ensemble is 4-D (lead at index 1);
        # deterministic is 3-D (lead at index 0). The member axis (if present)
        # is full-extent so it contributes no additional chunk dimension.
        lead_dim = 1 if len(shape) == 4 else 0
        n_spatial = len(shape) - lead_dim - 1
        # Number of spatial chunks along the trailing axes.
        spatial_chunk_counts = [
            (shape[lead_dim + 1 + i] + chunks[lead_dim + 1 + i] - 1)
            // chunks[lead_dim + 1 + i]
            for i in range(n_spatial)
        ]
        # Enumerate every spatial chunk combination for the target lead chunk.
        import itertools

        for combo in itertools.product(*[range(c) for c in spatial_chunk_counts]):
            coords = [0] * len(shape)
            coords[lead_dim] = lead_index
            coords[lead_dim + 1 :] = list(combo)
            key = array_path + "/" + ".".join(str(c) for c in coords)
            out.append(key)
    return sorted(set(out))


def build_object_inventory(store_path: str, array_paths: list[str]) -> set[str]:
    """Build an in-memory set of existing physical object keys for the given
    data-variable prefixes, using one paginated LIST per prefix.

    This avoids issuing a remote exists/HEAD per object during finalization.
    """
    inventory: set[str] = set()
    for array_path in array_paths:
        inventory.update(list_object_keys(store_path, array_path))
    return inventory


def validate_marker_evidence(
    store_path: str,
    *,
    marker_required_materialized: list[str],
    marker_omitted: list[str],
    actual_expected_keys: list[str],
    marker_expected_fingerprint: str,
    existing_objects: set[str] | None = None,
) -> None:
    """Validate a COMPLETE marker's completion evidence structurally.

    Args:
        store_path: The store path/URL.
        marker_required_materialized: The marker's required-materialized keys.
        marker_omitted: The marker's intentionally-omitted fill chunks.
        actual_expected_keys: The actual expected write set (from the store's
            .zarray chunk metadata + the region).
        marker_expected_fingerprint: The marker's recorded expected-write-set
            fingerprint.
        existing_objects: An in-memory set of existing physical object keys
            (built once per finalizer run via ``build_object_inventory``). When
            provided, existence is checked against this set (one LIST per
            prefix) instead of a remote exists per object. When omitted, a
            remote ``object_exists`` is issued per required key.

    Raises:
        InventoryError: If any structural invariant fails or a required
            materialized object is missing.
    """
    materialized = set(marker_required_materialized)
    omitted = set(marker_omitted)
    expected = set(actual_expected_keys)

    # 1. materialized set is a subset of the expected write set.
    unexpected_materialized = materialized - expected
    if unexpected_materialized:
        raise InventoryError(
            f"marker required-materialized keys not in the expected write set: "
            f"{sorted(unexpected_materialized)}"
        )
    # 2. omitted set is a subset of the expected write set.
    unexpected_omitted = omitted - expected
    if unexpected_omitted:
        raise InventoryError(
            f"marker omitted-fill keys not in the expected write set: "
            f"{sorted(unexpected_omitted)}"
        )
    # 3. the two sets are disjoint.
    overlap = materialized & omitted
    if overlap:
        raise InventoryError(
            f"marker materialized/omitted sets overlap: {sorted(overlap)}"
        )
    # 4. their union exactly covers the expected write set.
    union = materialized | omitted
    missing_from_union = expected - union
    if missing_from_union:
        raise InventoryError(
            f"marker materialized+omitted union does not cover the expected "
            f"write set; missing: {sorted(missing_from_union)}"
        )
    extra_in_union = union - expected
    if extra_in_union:
        raise InventoryError(
            f"marker union contains keys outside the expected write set: "
            f"{sorted(extra_in_union)}"
        )
    # 5. the marker's expected-write-set fingerprint matches the actual set.
    computed = expected_write_set_fingerprint(list(materialized), list(omitted))
    if computed != marker_expected_fingerprint:
        raise InventoryError(
            "marker expected-write-set fingerprint does not match the materialized "
            "+ omitted sets"
        )
    # 6. every required materialized object currently exists.
    for key in sorted(materialized):
        exists = (
            key in existing_objects
            if existing_objects is not None
            else object_exists(store_path, key)
        )
        if not exists:
            raise InventoryError(
                f"required materialized object is missing: {key!r} (external "
                "deletion invalidates committed state)"
            )
    # 7. intentionally omitted fill chunks may be physically absent (no check).
