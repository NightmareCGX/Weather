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
        use_listings_cache=False,
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


def _read_zattrs(store_path: str, array_path: str) -> dict[str, object] | None:
    """Read a data variable's ``.zattrs`` metadata, or None when absent."""
    backend, root = _storage_backend(store_path)
    rel = f"{array_path}/.zattrs"
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


def _array_dimension_layout(
    store_path: str,
    array_path: str,
    shape: list[int],
    *,
    zattrs_cache: dict[str, dict[str, object]] | None = None,
) -> tuple[bool, int | None, int, list[int]]:
    """Determine the semantic dimension layout of a stored Zarr array.

    First inspects the array's ``.zattrs`` metadata for ``_ARRAY_DIMENSIONS``
    to resolve exact named axis positions. Falls back to the canonical platform
    contract where 4-D represents ``(member, lead_time_hours, latitude, longitude)``
    and 3-D represents ``(lead_time_hours, latitude, longitude)``.

    Returns:
        A tuple of ``(has_member, member_dim, lead_dim, spatial_dims)``.
    """
    za_attrs: dict[str, object] | None
    if zattrs_cache is not None and array_path in zattrs_cache:
        za_attrs = zattrs_cache[array_path]
    else:
        za_attrs = _read_zattrs(store_path, array_path)
        if za_attrs is not None and zattrs_cache is not None:
            zattrs_cache[array_path] = za_attrs

    if za_attrs is not None:
        dims_obj = za_attrs.get("_ARRAY_DIMENSIONS")
        if isinstance(dims_obj, list) and all(isinstance(d, str) for d in dims_obj):
            dims: list[str] = list(dims_obj)
            has_member = "member" in dims
            member_dim = dims.index("member") if has_member else None
            lead_dim = (
                dims.index("lead_time_hours")
                if "lead_time_hours" in dims
                else (1 if has_member else 0)
            )
            spatial_dims = [
                i for i in range(len(shape)) if i != member_dim and i != lead_dim
            ]
            return has_member, member_dim, lead_dim, spatial_dims

    # Fallback to positional contract: 4-D is (member, lead, lat, lon),
    # 3-D is (lead, lat, lon).
    has_member = len(shape) == 4
    member_dim = 0 if has_member else None
    lead_dim = 1 if has_member else 0
    spatial_dims = list(range(lead_dim + 1, len(shape)))
    return has_member, member_dim, lead_dim, spatial_dims


def _derive_region_chunk_keys(
    store_path: str,
    array_path: str,
    *,
    member: int | None,
    lead_index: int,
    za: dict[str, object],
    zattrs_cache: dict[str, dict[str, object]] | None = None,
) -> list[str]:
    """Derive the physical chunk keys a logical region maps to in one array."""
    shape_raw = za.get("shape")
    chunks_raw = za.get("chunks")
    if not isinstance(shape_raw, list) or not isinstance(chunks_raw, list):
        return []
    shape = [int(s) for s in shape_raw]
    chunks = [int(c) for c in chunks_raw]

    has_member, member_dim, lead_dim, spatial_dims = _array_dimension_layout(
        store_path, array_path, shape, zattrs_cache=zattrs_cache
    )

    member_chunk = 0
    if has_member and member is not None and member_dim is not None:
        member_index = _member_positional_index(store_path, array_path, member)
        member_chunk = (
            member_index // chunks[member_dim] if chunks[member_dim] > 0 else 0
        )

    spatial_chunk_counts = [
        (shape[d] + chunks[d] - 1) // chunks[d] for d in spatial_dims
    ]

    out: list[str] = []
    import itertools

    for combo in itertools.product(*[range(c) for c in spatial_chunk_counts]):
        coords = [0] * len(shape)
        if has_member and member_dim is not None:
            coords[member_dim] = member_chunk
        coords[lead_dim] = lead_index
        for i, d in enumerate(spatial_dims):
            coords[d] = combo[i]
        key = array_path + "/" + ".".join(str(c) for c in coords)
        out.append(key)
    return out


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

    Under the member_chunk=1 ensemble layout, different members at the same lead
    map to distinct member-chunk coordinates -> disjoint keys -> concurrent writes.
    Under legacy member_chunk=30 stores, different members map to member-chunk 0 ->
    identical conflict keys -> serialized writes.

    Args:
        store_path: The store path/URL.
        member: The ensemble member identity, or ``None`` for deterministic.
        lead_index: The positional index of the lead in the store's
            ``lead_time_hours`` coordinate.
        data_var_paths: The data-variable array paths.

    Returns:
        A sorted, de-duplicated list of physical-chunk conflict identities
        (e.g. ``"temperature_2m/0.0.0.0"`` for a member_chunk=1 chunk).
    """
    out: list[str] = []
    for array_path in data_var_paths:
        za = _read_zarray(store_path, array_path)
        if za is None:
            continue
        out.extend(
            _derive_region_chunk_keys(
                store_path, array_path, member=member, lead_index=lead_index, za=za
            )
        )
    return sorted(set(out))


def region_expected_object_keys(
    store_path: str,
    *,
    member: int | None,
    lead_index: int,
    data_var_paths: list[str],
    zarray_cache: dict[str, dict[str, object]] | None = None,
    zattrs_cache: dict[str, dict[str, object]] | None = None,
) -> list[str]:
    """Derive the physical chunk object keys a logical region writes.

    The actual store's ``.zarray`` chunk metadata determines the physical chunk
    grid. Uses the exact same chunk-coordinate derivation as
    :func:`physical_conflict_keys`, ensuring complete consistency between
    advisory locking and marker completion evidence.

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
        zattrs_cache: An optional in-memory cache of per-array ``.zattrs``
            metadata.

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
        out.extend(
            _derive_region_chunk_keys(
                store_path,
                array_path,
                member=member,
                lead_index=lead_index,
                za=za,
                zattrs_cache=zattrs_cache,
            )
        )
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
