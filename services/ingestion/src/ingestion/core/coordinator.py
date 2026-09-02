"""Region-write concurrency coordinator for the ingestion worker.

This module implements the approved concurrency protocol on top of the
foundational primitives:

* ``ingestion.core.locks`` — PostgreSQL advisory-lock coordinator;
* ``ingestion.core.markers`` — stable per-region markers + protocol version;
* ``ingestion.core.marker_put_scheduler`` — rolling bounded marker PUTs;
* ``ingestion.core.cancel`` — non-abandoning aggregate drain;
* ``domain.locks`` / ``domain.serving`` — pure identity/fingerprint helpers.

The coordinator is a thin orchestration layer the CLI calls. It keeps the
existing ``ingest_grib_file`` library path intact (which is restricted to
non-live stores); the CLI uses the coordinator for the live, locked path.

Protocol (frozen in Checkpoint 2H):

    admission turnstile
        → store gate (SHARED for writers/readers, EXCLUSIVE for init/finalize)
        → sorted unique physical-region locks

Data commit order:

    UPDATING marker declared
        → region data objects
        → COMPLETE marker
        → committed manifest
        → catalog reconciliation/status commit
"""

from __future__ import annotations

import logging
import threading
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from domain.locks import (
    logical_region_encoding,
    sha256_hex,
    serving_state_fingerprint,
)
from ingestion.core.config import settings
from ingestion.core.base import (
    StoreSchemaMismatchError,
    is_retryable_storage_error,
)
from ingestion.core.catalog import (
    CommittedState,
    ModelRunRecord,
    RunCatalogSpec,
    set_run_partial,
)
from ingestion.core.locks import StoreLockCoordinator
from ingestion.core.markers import (
    HYBRID,
    LEGACY,
    MARKER_V1,
    list_region_marker_keys,
    read_manifest,
    read_protocol_version,
    read_region_marker,
    region_evidence_fingerprint,
    write_manifest,
    write_protocol_version,
    write_region_marker,
)
from ingestion.core.pipeline import (
    _commit_region,
    guard_full_overwrite,
)
from ingestion.core.zarr_writer import (
    prepare_run_store,
    store_exists,
)

logger = logging.getLogger(__name__)


class WavePreUpdateError(RuntimeError):
    """Raised when the wave pre-update fails (no data worker starts)."""


@dataclass
class WaveRegion:
    """A logical region target of a wave."""

    lead_time_hours: int
    member: int | None
    generation: str

    @property
    def region_id(self) -> str:
        return logical_region_encoding(
            lead_time_hours=self.lead_time_hours, member=self.member
        )


@dataclass
class Wave:
    """A bounded wave of target regions."""

    regions: list[WaveRegion] = field(default_factory=list)

    def region_ids(self) -> list[str]:
        return sorted({r.region_id for r in self.regions})


@dataclass(frozen=True)
class FinalizeResult:
    """The authoritative result of a coalesced finalization run.

    Attributes:
        status: The derived run status ('ready', 'partial', or 'processing').
        committed_regions: Mapping of logical region id (e.g. 'det_L0006',
            'mem017_L0006') to the committed generation UUID for all regions
            durably committed and reconciled in this run.
    """

    status: str
    committed_regions: Mapping[str, str] = field(default_factory=dict)


def _new_generation() -> str:
    import uuid

    return uuid.uuid4().hex


def _physical_conflict_region_ids(
    dataset: xr.Dataset, store_path: str, *, member: int | None
) -> list[str]:
    """Derive the physical-conflict region ids for a commit.

    The conflict identities come from the store's ACTUAL ``.zarray`` chunk
    geometry via :func:`ingestion.core.inventory.physical_conflict_keys`,
    mapping the logical region ``(member?, lead, lat, lon)`` to every physical
    chunk coordinate it can modify.

    Under the current ensemble layout (member full-extent, lead chunked at 1),
    different members at the same lead map to the SAME member chunk and lead
    chunk -> overlapping conflict keys -> they serialize. Different leads map
    to different lead chunks -> disjoint keys -> concurrent.

    For a layout where member is chunked at 1, different members map to
    different member-chunk coordinates -> distinct keys.
    """
    from ingestion.core.inventory import physical_conflict_keys

    lead_values = dataset.coords["lead_time_hours"].values
    lead = int(np.asarray(lead_values).reshape(-1)[0])
    lead_index = _lead_index_in_store(store_path, lead)
    data_var_paths = sorted(str(v) for v in dataset.data_vars)
    return physical_conflict_keys(
        store_path, member=member, lead_index=lead_index, data_var_paths=data_var_paths
    )


@dataclass(frozen=True)
class StoreMetadataSnapshot:
    """Immutable, generation-bound metadata snapshot of an initialized Zarr store.

    Captures coordinate indices, variable schemas, and .zarray/.zattrs physical chunk
    geometries so that region-write workers can perform validation, conflict derivation,
    and slice resolution in-memory without remote store re-opens.
    """

    store_path: str
    generation: str | None
    is_ensemble: bool
    data_var_paths: tuple[str, ...]
    lead_index_map: Mapping[int, int]
    member_index_map: Mapping[int, int]
    zarray_by_var: Mapping[str, Mapping[str, object]]
    zattrs_by_var: Mapping[str, Mapping[str, object]]
    data_var_dims: Mapping[str, tuple[str, ...]]
    coords_values: Mapping[str, tuple[float, ...]]
    grid_shape: tuple[int, int]
    cycle_time: str | None
    model_id: str | None


class RunCoordinator:
    """Coordinates the wave pre-update, region workers, and coalesced finalizer.

    One coordinator instance is created per CLI run. It owns the store-gate and
    admission keys (derived from the canonical store identity) and dispatches
    the lock protocol.
    """

    def __init__(
        self,
        spec: RunCatalogSpec,
        store_path: str,
        *,
        endpoint: str | None = None,
        secure: bool = False,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.spec = spec
        self.store_path = store_path
        self.endpoint = endpoint
        self.secure = secure
        self.timeout_seconds = timeout_seconds
        # Cache the lead -> positional-index map (read once per finalizer run).
        self._lead_index_cache: dict[int, int] = {}
        # Cache the data-variable .zarray and .zattrs geometry per array path.
        self._zarray_cache: dict[str, dict[str, object]] = {}
        self._zattrs_cache: dict[str, dict[str, object]] = {}
        # Cache member coordinate positional indices.
        self._member_index_cache: dict[int, int] = {}
        # Immutable metadata snapshot of the store built under the exclusive gate.
        self._snapshot: StoreMetadataSnapshot | None = None

    def _lead_index_for(self, lead: int) -> int:
        """Return the positional lead index, caching per lead value."""
        if self._snapshot is not None and lead in self._snapshot.lead_index_map:
            return self._snapshot.lead_index_map[lead]
        if lead not in self._lead_index_cache:
            self._lead_index_cache.update(_load_lead_indices_in_store(self.store_path))
        if lead not in self._lead_index_cache:
            raise ValueError(f"lead {lead} not found in store {self.store_path!r}")
        return self._lead_index_cache[lead]

    def _build_snapshot(self) -> StoreMetadataSnapshot:
        """Build an immutable snapshot from the store's persisted metadata."""
        from ingestion.core.inventory import _read_zarray, _read_zattrs
        from ingestion.core.pipeline import _resolve_cycle_time
        from ingestion.core.markers import read_manifest
        from ingestion.core.zarr_writer import _resolve_store

        resolved = _resolve_store(self.store_path)
        ds = xr.open_zarr(resolved, consolidated=False)
        manifest = read_manifest(self.store_path)
        gen = str(manifest.get("generation")) if manifest and manifest.get("generation") else None

        lead_values = np.atleast_1d(ds.coords["lead_time_hours"].values).reshape(-1)
        lead_index_map = {int(v): i for i, v in enumerate(lead_values)}
        member_index_map = {}
        if "member" in ds.coords:
            member_values = np.atleast_1d(ds.coords["member"].values).reshape(-1)
            member_index_map = {int(v): i for i, v in enumerate(member_values)}

        data_var_paths = tuple(sorted(str(v) for v in ds.data_vars))
        zarray_by_var: dict[str, Mapping[str, object]] = {}
        zattrs_by_var: dict[str, Mapping[str, object]] = {}
        data_var_dims: dict[str, tuple[str, ...]] = {}
        for var in data_var_paths:
            za = _read_zarray(self.store_path, var)
            if za is not None:
                zarray_by_var[var] = za
                self._zarray_cache[var] = za
            zat = _read_zattrs(self.store_path, var)
            if zat is not None:
                zattrs_by_var[var] = zat
            data_var_dims[var] = tuple(str(d) for d in ds[var].dims if str(d) != "member")

        coords_values: dict[str, tuple[float, ...]] = {}
        for axis in ("latitude", "longitude"):
            if axis in ds.coords:
                coords_values[axis] = tuple(float(v) for v in ds.coords[axis].values)

        lat_len = ds.sizes.get("latitude", 0)
        lon_len = ds.sizes.get("longitude", 0)
        grid_shape = (int(lat_len), int(lon_len))
        cycle_time = _resolve_cycle_time(ds)
        model_id = str(ds.attrs.get("model_id")) if "model_id" in ds.attrs else None
        ds.close()

        return StoreMetadataSnapshot(
            store_path=self.store_path,
            generation=gen,
            is_ensemble=self.spec.is_ensemble,
            data_var_paths=data_var_paths,
            lead_index_map=lead_index_map,
            member_index_map=member_index_map,
            zarray_by_var=zarray_by_var,
            zattrs_by_var=zattrs_by_var,
            data_var_dims=data_var_dims,
            coords_values=coords_values,
            grid_shape=grid_shape,
            cycle_time=cycle_time,
            model_id=model_id,
        )

    # ------------------------------------------------------------------
    # Retained-seed initialization (Step 5)
    # ------------------------------------------------------------------
    def initialize_run_store(
        self,
        conn: Connection,
        *,
        seed_dataset: xr.Dataset,
        expected_leads: tuple[int, ...],
        expected_members: tuple[int, ...],
        run_id: str | None,
        is_same_cycle: bool,
        observer: object | None = None,
    ) -> None:
        """Initialize the store under the EXCLUSIVE gate (retained-seed flow).

        Args:
            conn: The worker's physical Connection (holds the EXCLUSIVE gate).
            seed_dataset: The retained parsed/normalized seed dataset.
            expected_leads: The run's expected lead set.
            expected_members: The run's expected member set (empty for
                deterministic).
            run_id: The existing run id when this is a same-cycle re-ingest
                (downgrade to partial), else ``None`` (fresh run creation is
                handled by the finalizer).
            is_same_cycle: Whether this wave re-ingests an existing live run.
            observer: Optional progress/milestone observer.

        Raises:
            LiveStoreOverwriteError: If a live run owns the absent store path.
        """
        co = StoreLockCoordinator(
            conn,
            store_path=self.store_path,
            endpoint=self.endpoint,
            secure=self.secure,
            timeout_seconds=self.timeout_seconds,
        )
        if observer is not None and hasattr(observer, "record_milestone"):
            observer.record_milestone("store_gate_wait_start")
        co.acquire_exclusive_gate()
        if observer is not None and hasattr(observer, "record_milestone"):
            observer.record_milestone("store_gate_acquired")
        try:
            if store_exists(self.store_path):
                # Existing store: validate identity only; the region worker's
                # _commit_region performs schema validation after expanding the
                # lead/member dims (the raw seed is 2-D (lat, lon) until then).
                from ingestion.core.zarr_writer import _resolve_store

                resolved = _resolve_store(self.store_path)
                existing = xr.open_zarr(resolved, consolidated=False)
                _validate_store_identity(seed_dataset, existing, self.store_path)
                existing.close()
                if run_id is not None and is_same_cycle:
                    with Session(bind=conn) as db:
                        set_run_partial(db, run_id)
                        db.commit()
                self._snapshot = self._build_snapshot()
                return
            # Absent store: guard against a live-owned path, then initialize.
            with Session(bind=conn) as db:
                guard_full_overwrite(db, self.store_path)
                db.commit()
            if observer is not None and hasattr(observer, "set_init_phase"):
                observer.set_init_phase("prepare_run_store")
            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("prepare_run_store_start")
            prepare_run_store(
                seed_dataset,
                self.store_path,
                expected_lead_time_hours=expected_leads,
                expected_members=expected_members,
            )
            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("prepare_run_store_complete")
            # New stores are strict marker_v1.
            write_protocol_version(self.store_path, MARKER_V1)
            self._snapshot = self._build_snapshot()
        finally:
            co.release_exclusive_gate()

    # ------------------------------------------------------------------
    # Wave pre-update (Steps 6 & 7)
    # ------------------------------------------------------------------
    def pre_update_wave(
        self,
        conn: Connection,
        *,
        regions: list[WaveRegion],
        run_id: str | None,
        is_same_cycle: bool,
        executor: object,
        cancel_event: threading.Event,
        observer: object | None = None,
    ) -> list[WaveRegion]:
        """Declare UPDATING markers for every target under the EXCLUSIVE gate.

        The run is downgraded to ``partial`` (minimal downgrade) before any
        data mutation. Every target UPDATING marker is written with the rolling
        bounded scheduler. Returns the regions with their generations.

        Raises:
            WavePreUpdateError: If any marker PUT failed/cancelled (no data
                worker may start).
        """
        from concurrent.futures import ThreadPoolExecutor

        assert isinstance(executor, ThreadPoolExecutor)
        co = StoreLockCoordinator(
            conn,
            store_path=self.store_path,
            endpoint=self.endpoint,
            secure=self.secure,
            timeout_seconds=self.timeout_seconds,
        )
        co.acquire_admission()
        co.acquire_exclusive_gate()
        try:
            if run_id is not None and is_same_cycle:
                with Session(bind=conn) as db:
                    set_run_partial(db, run_id)
                    db.commit()

            if self._snapshot is None:
                self._snapshot = self._build_snapshot()

            # Allocate a fresh generation per target and write UPDATING markers.
            from ingestion.core.marker_put_scheduler import put_markers_rolling

            if observer is not None and hasattr(observer, "set_init_phase"):
                observer.set_init_phase("pre_update_markers")
            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("pre_update_start")

            targets = {r.region_id: r for r in regions}
            region_ids = sorted(targets)
            put_one = self._make_updating_put(regions)
            result = put_markers_rolling(
                region_ids,
                put_one,
                concurrency=min(8, max(1, len(region_ids))),
                cancel_event=cancel_event,
                timeout_seconds=self.timeout_seconds,
                executor=executor,
                observer=observer,
            )
            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("pre_update_complete")

            if not result.ok:
                raise WavePreUpdateError(
                    f"wave pre-update failed: {len(result.failures)} PUT failures, "
                    f"{len(result.cancelled)} cancelled; no data worker started"
                )
            return regions
        finally:
            co.release_exclusive_gate()
            co.release_admission()

    def _make_updating_put(self, regions: list[WaveRegion]) -> "Callable[[str], None]":
        """Build the per-region UPDATING marker PUT callable."""

        def put_one(region_id: str) -> None:
            for r in regions:
                if r.region_id == region_id:
                    write_region_marker(
                        self.store_path,
                        lead_time_hours=r.lead_time_hours,
                        member=r.member,
                        payload={
                            "protocol_version": 1,
                            "state": "updating",
                            "generation": r.generation,
                            "logical_region": {
                                "lead_time_hours": r.lead_time_hours,
                                **(
                                    {"member": r.member} if r.member is not None else {}
                                ),
                            },
                            "expected_write_set_fingerprint": "",
                            "required_materialized_object_keys": [],
                            "intentionally_omitted_fill_chunks": [],
                        },
                    )
                    return
            raise ValueError(f"unknown region id {region_id!r}")

        return put_one

    # ------------------------------------------------------------------
    # Region-write worker (Step 8)
    # ------------------------------------------------------------------
    def write_region_worker(
        self,
        conn: Connection,
        *,
        dataset: xr.Dataset,
        member: int | None,
        generation: str,
        expected_leads: tuple[int, ...],
        expected_members: tuple[int, ...],
    ) -> None:
        """Write one region under the SHARED gate + region locks.

        The generation-ownership check (marker.state == UPDATING AND
        marker.generation == worker_generation) occurs AFTER region-lock
        acquisition and BEFORE the first data-object mutation. A mismatch
        aborts with ZERO data writes.
        """
        co = StoreLockCoordinator(
            conn,
            store_path=self.store_path,
            endpoint=self.endpoint,
            secure=self.secure,
            timeout_seconds=self.timeout_seconds,
        )
        co.acquire_shared_admission()
        co.acquire_shared_gate()
        try:
            snapshot = self._snapshot or self._build_snapshot()
            lead_values = dataset.coords["lead_time_hours"].values
            lead = int(np.asarray(lead_values).reshape(-1)[0])
            lead_index = snapshot.lead_index_map.get(lead)
            if lead_index is None:
                raise StoreSchemaMismatchError(f"lead {lead} not found in store coordinate")

            from ingestion.core.inventory import physical_conflict_keys

            region_ids = physical_conflict_keys(
                self.store_path,
                member=member,
                lead_index=lead_index,
                data_var_paths=snapshot.data_var_paths,
                zarray_cache=snapshot.zarray_by_var,
                zattrs_cache=snapshot.zattrs_by_var,
                member_index_cache=snapshot.member_index_map,
            )
            co.acquire_region_locks(region_ids)
            try:
                marker = read_region_marker(
                    self.store_path, lead_time_hours=lead, member=member
                )
                if (
                    marker.get("state") != "updating"
                    or marker.get("generation") != generation
                ):
                    logger.error(
                        "region %s generation mismatch: expected generation %s, "
                        "marker=%s; aborting with zero data writes",
                        (member, lead),
                        generation,
                        marker.get("generation"),
                    )
                    raise StoreSchemaMismatchError(
                        f"region (member={member}, lead={lead}) is not owned by "
                        f"generation {generation}"
                    )
                # Generation-ownership confirmed: bounded retry for transient storage writes
                from ingestion.core.inventory import (
                    expected_write_set_fingerprint,
                    region_expected_object_keys,
                    verify_expected_object_keys,
                )

                max_attempts = 3
                base_delay = 0.2
                max_delay = 2.0

                for attempt in range(1, max_attempts + 1):
                    try:
                        # 1. Write the data region using snapshot
                        _commit_region(
                            dataset,
                            self.store_path,
                            member=member,
                            expected_lead_time_hours=expected_leads,
                            expected_members=expected_members,
                            snapshot=snapshot,
                        )
                        # 2. Compute physical object inventory for COMPLETE marker from snapshot
                        expected_keys = region_expected_object_keys(
                            self.store_path,
                            member=member,
                            lead_index=lead_index,
                            lead_time_hours=lead,
                            format_version=getattr(settings, "STORAGE_FORMAT_VERSION", "sharded_v1"),
                            data_var_paths=snapshot.data_var_paths,
                            zarray_cache=snapshot.zarray_by_var,
                            zattrs_cache=snapshot.zattrs_by_var,
                            member_index_cache=snapshot.member_index_map,
                        )
                        existing_keys = verify_expected_object_keys(
                            self.store_path,
                            expected_keys,
                            member=member,
                            lead_index=lead_index,
                            zarray_cache=snapshot.zarray_by_var,
                            zattrs_cache=snapshot.zattrs_by_var,
                            member_index_cache=snapshot.member_index_map,
                        )
                        # Real-data writes materialize all expected chunks; an expected
                        # chunk that is absent is an all-fill omission.
                        required = [k for k in expected_keys if k in existing_keys]
                        omitted = [k for k in expected_keys if k not in existing_keys]
                        # 3. Write COMPLETE marker (last store-side operation).
                        write_region_marker(
                            self.store_path,
                            lead_time_hours=lead,
                            member=member,
                            payload={
                                "protocol_version": 1,
                                "state": "complete",
                                "generation": generation,
                                "logical_region": {
                                    "lead_time_hours": lead,
                                    **({"member": member} if member is not None else {}),
                                },
                                "expected_write_set_fingerprint": expected_write_set_fingerprint(
                                    required, omitted
                                ),
                                "required_materialized_object_keys": required,
                                "intentionally_omitted_fill_chunks": omitted,
                            },
                        )
                        if attempt > 1:
                            logger.info(
                                "Region write succeeded on retry: member=%s lead=%d attempt=%d/%d",
                                member,
                                lead,
                                attempt,
                                max_attempts,
                            )
                        break
                    except Exception as exc:
                        if attempt < max_attempts and is_retryable_storage_error(exc):
                            jitter = random.uniform(0.0, 0.1)
                            backoff = min(max_delay, base_delay * (2 ** (attempt - 1))) + jitter
                            logger.warning(
                                "Transient storage failure on region write (member=%s lead=%d attempt=%d/%d): %s; "
                                "retrying in %.2fs",
                                member,
                                lead,
                                attempt,
                                max_attempts,
                                exc,
                                backoff,
                            )
                            time.sleep(backoff)
                            continue
                        logger.error(
                            "Region write failed (member=%s lead=%d attempt=%d/%d retryable=%s): %s",
                            member,
                            lead,
                            attempt,
                            max_attempts,
                            is_retryable_storage_error(exc),
                            exc,
                        )
                        raise
            finally:
                co.release_region_locks(region_ids)
        finally:
            co.release_shared_gate()
            co.release_shared_admission()

    # ------------------------------------------------------------------
    # Coalesced finalization (Step 10)
    # ------------------------------------------------------------------
    def finalize_run(
        self,
        conn: Connection,
        *,
        run_id: str,
        spec: RunCatalogSpec,
        expected_leads: tuple[int, ...],
        expected_members: tuple[int, ...],
        observer: object | None = None,
        marker_concurrency: int | None = None,
        verify_full_inventory: bool = False,
    ) -> FinalizeResult:
        """Run the single coalesced finalization for the bounded CLI wave.

        In normal realtime ingestion (verify_full_inventory=False), operates in
        O(regions) complexity on marker evidence without scanning physical Zarr chunk
        objects.

        Returns the authoritative FinalizeResult with status and committed_regions.
        """
        co = StoreLockCoordinator(
            conn,
            store_path=self.store_path,
            endpoint=self.endpoint,
            secure=self.secure,
            timeout_seconds=self.timeout_seconds,
        )
        if observer is not None and hasattr(observer, "record_milestone"):
            observer.record_milestone("finalize_start")
        co.acquire_admission()
        co.acquire_exclusive_gate()
        try:
            mode = read_protocol_version(self.store_path)

            # Phase 1: Marker Listing
            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("marker_listing_start")
            marker_keys = list_region_marker_keys(self.store_path)
            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("marker_listing_complete")

            # Phase 2: Marker Read & Validation
            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("marker_read_validation_start")

            data_var_paths = _store_data_var_paths(self.store_path, snapshot=self._snapshot)
            existing_objects: set[str] | None = None
            if verify_full_inventory:
                from ingestion.core.inventory import build_object_inventory

                existing_objects = (
                    build_object_inventory(self.store_path, data_var_paths)
                    if data_var_paths
                    else set()
                )

            # Populate the .zarray, .zattrs, and member coordinate caches once
            # so per-marker region-key derivation avoids repeated remote reads.
            from ingestion.core.inventory import _read_zarray, _read_zattrs

            for array_path in data_var_paths:
                if array_path not in self._zarray_cache:
                    za = _read_zarray(self.store_path, array_path)
                    if za is not None:
                        self._zarray_cache[array_path] = za
                if array_path not in self._zattrs_cache:
                    zat = _read_zattrs(self.store_path, array_path)
                    if zat is not None:
                        self._zattrs_cache[array_path] = zat

            if not self._member_index_cache and spec.is_ensemble:
                if self._snapshot is not None and self._snapshot.member_index_map:
                    self._member_index_cache = dict(self._snapshot.member_index_map)
                else:
                    try:
                        from ingestion.core.zarr_writer import _resolve_store

                        resolved = _resolve_store(self.store_path)
                        ds = xr.open_zarr(resolved, consolidated=False)
                        if "member" in ds.coords:
                            member_vals = np.atleast_1d(ds.coords["member"].values).reshape(-1)
                            self._member_index_cache = {int(v): i for i, v in enumerate(member_vals)}
                        ds.close()
                    except Exception:
                        pass

            committed: dict[str, str] = {}  # region_id -> generation
            updating: list[str] = []
            marker_results = _read_marker_payloads_bounded(
                self.store_path, marker_keys, max_concurrency=marker_concurrency
            )
            for key, payload in marker_results:
                region_id = key.rsplit("/", 1)[-1].removesuffix(".json")
                state = payload.get("state")
                gen = payload.get("generation")
                if state == "complete":
                    if self._marker_evidence_valid(
                        region_id,
                        payload,
                        existing_objects=existing_objects,
                        verify_physical_objects=verify_full_inventory,
                    ):
                        committed[region_id] = str(gen)
                    else:
                        updating.append(region_id)
                elif state == "updating":
                    updating.append(region_id)

            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("marker_read_validation_complete")

            # Phase 3: Manifest Generation & Write
            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("manifest_write_start")
                observer.record_milestone("manifest_payload_build_start")

            # Hybrid mode: marker-less regions use the legacy rule.
            legacy_evidence: list[str] = []
            if mode == HYBRID:
                # Any region in the store's expected set without a marker is a
                # legacy region (kept under the legacy committed-state rule).
                pass
            committed_state = self._committed_state_from_regions(
                committed, updating, expected_leads, expected_members, mode
            )
            # Compute fingerprints + manifest generation.
            run_identity = {
                "model_version_id": spec.version_string,
                "cycle_time": spec.cycle_time.isoformat(),
                "is_ensemble": spec.is_ensemble,
            }
            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("manifest_fingerprint_start")
            store_schema_fp = _store_schema_fingerprint(self.store_path, snapshot=self._snapshot)
            legacy_fp = (
                region_evidence_fingerprint(self.store_path, legacy_evidence)
                if mode in (LEGACY, HYBRID)
                else None
            )
            serving_fp = serving_state_fingerprint(
                store_protocol_mode=mode,
                run_identity=run_identity,
                store_schema_fingerprint=store_schema_fp,
                region_serving_states=_region_serving_states(committed, updating, mode),
            )
            committed_fp = sha256_hex("committed", *sorted(committed.keys()))
            markers_fp = sha256_hex("markers", *sorted(marker_keys))
            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("manifest_fingerprint_complete")

            existing_manifest = read_manifest(self.store_path)
            if (
                existing_manifest is not None
                and existing_manifest.get("serving_state_fingerprint") == serving_fp
            ):
                generation = str(existing_manifest.get("generation"))
            else:
                generation = _new_generation()

            payload = {
                "manifest_schema_version": 1,
                "store_protocol_mode": mode,
                "storage_format_version": getattr(settings, "STORAGE_FORMAT_VERSION", "sharded_v1"),
                "generation": generation,
                "run_identity": run_identity,
                "canonical_store_identity_hash": _store_identity_hash(self.store_path),
                "serving_state_fingerprint": serving_fp,
                "committed_state_fingerprint": committed_fp,
                "store_schema_fingerprint": store_schema_fp,
                "region_marker_set_fingerprint": markers_fp,
                "legacy_region_evidence_fingerprint": legacy_fp,
            }
            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("manifest_payload_build_complete")
                observer.record_milestone("manifest_put_start")

            write_manifest(self.store_path, payload)

            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("manifest_put_complete")
                observer.record_milestone("manifest_write_complete")

            # Phase 4: Catalog Reconciliation & Status Commit
            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("catalog_reconcile_start")

            with Session(bind=conn) as db:
                run = db.get(ModelRunRecord, run_id)
                if run is None:
                    raise RuntimeError(f"run {run_id} not found during finalization")
                from ingestion.core.catalog import (
                    _reconcile_catalog_to_store,
                    _derive_run_status,
                )

                _reconcile_catalog_to_store(db, run, committed_state, spec)
                status = _derive_run_status(db, run, spec, committed_state)
                setattr(run, "status", status)
                db.commit()

            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("catalog_reconcile_complete")

            if observer is not None and hasattr(observer, "record_milestone"):
                observer.record_milestone("finalize_complete")
            return FinalizeResult(status=status, committed_regions=committed)
        finally:
            co.release_exclusive_gate()
            co.release_admission()

    def publish_settled_lead(
        self,
        conn: Connection,
        *,
        run_id: str,
        spec: RunCatalogSpec,
        lead_time_hours: int,
        expected_members: tuple[int, ...],
    ) -> None:
        """Publish a settled forecast lead to the catalog and advance serving generation.

        Executed after all expected member tasks for a specific lead have settled.
        Reads COMPLETE markers for that lead, reconciles catalog rows (forecast_products
        and ensemble_member_products), updates manifest.json with a new serving generation,
        and commits the database transaction.

        Does NOT mark the overall run status as 'ready' (status remains 'processing' or 'partial').
        """
        co = StoreLockCoordinator(
            conn,
            store_path=self.store_path,
            endpoint=self.endpoint,
            secure=self.secure,
            timeout_seconds=self.timeout_seconds,
        )
        co.acquire_admission()
        co.acquire_exclusive_gate()
        try:
            mode = read_protocol_version(self.store_path)
            committed_for_lead: set[int] = set()
            if spec.is_ensemble:
                members_to_check = expected_members if expected_members else tuple(range(1, 31))
                candidate_keys = [
                    f".markers/regions/mem{m:03d}_L{lead_time_hours:04d}.json"
                    for m in members_to_check
                ]
                marker_results = _read_marker_payloads_bounded(
                    self.store_path, candidate_keys, max_concurrency=16
                )
                for key, payload in marker_results:
                    region_id = key.rsplit("/", 1)[-1].removesuffix(".json")
                    if isinstance(payload, Mapping) and payload.get("state") == "complete" and self._marker_evidence_valid(
                        region_id, payload
                    ):
                        logical_reg = payload.get("logical_region")
                        if isinstance(logical_reg, Mapping):
                            m_val = logical_reg.get("member")
                            if m_val is not None:
                                committed_for_lead.add(int(str(m_val)))
            else:
                candidate_keys = [f".markers/regions/det_L{lead_time_hours:04d}.json"]
                marker_results = _read_marker_payloads_bounded(
                    self.store_path, candidate_keys, max_concurrency=1
                )
                for key, payload in marker_results:
                    region_id = key.rsplit("/", 1)[-1].removesuffix(".json")
                    if isinstance(payload, Mapping) and payload.get("state") == "complete" and self._marker_evidence_valid(
                        region_id, payload
                    ):
                        committed_for_lead.add(0)

            with Session(bind=conn) as db:
                run = db.get(ModelRunRecord, run_id)
                if run is None:
                    return

                from ingestion.core.catalog import (
                    EnsembleMemberProductRecord,
                    EnsembleMemberRecord,
                    ProductRecord,
                    _get_or_create,
                )

                is_postgres = bool(db.bind and db.bind.dialect.name == "postgresql")

                if spec.is_ensemble and committed_for_lead:
                    if is_postgres:
                        from sqlalchemy.dialects.postgresql import insert as pg_insert

                        member_values = [
                            {
                                "id": f"member_{member_num}_{run.id}",
                                "run_id": run.id,
                                "member_index": member_num,
                                "member_name": f"{spec.model_id}_member_{member_num}",
                            }
                            for member_num in sorted(committed_for_lead)
                        ]
                        stmt_mem = pg_insert(EnsembleMemberRecord).values(member_values)
                        stmt_mem = stmt_mem.on_conflict_do_nothing(
                            index_elements=[
                                EnsembleMemberRecord.run_id,
                                EnsembleMemberRecord.member_index,
                            ]
                        )
                        db.execute(stmt_mem)

                        member_prod_values = [
                            {
                                "id": f"member_product_{member_num}_{lead_time_hours}_{run.id}",
                                "run_id": run.id,
                                "member_index": member_num,
                                "lead_time_hours": lead_time_hours,
                            }
                            for member_num in sorted(committed_for_lead)
                        ]
                        stmt_mprod = pg_insert(EnsembleMemberProductRecord).values(member_prod_values)
                        stmt_mprod = stmt_mprod.on_conflict_do_nothing(
                            index_elements=[
                                EnsembleMemberProductRecord.run_id,
                                EnsembleMemberProductRecord.member_index,
                                EnsembleMemberProductRecord.lead_time_hours,
                            ]
                        )
                        db.execute(stmt_mprod)
                    else:
                        for member_num in sorted(committed_for_lead):
                            _get_or_create(
                                db,
                                EnsembleMemberRecord,
                                (EnsembleMemberRecord.run_id == run.id)
                                & (EnsembleMemberRecord.member_index == member_num),
                                {
                                    "id": f"member_{member_num}_{run.id}",
                                    "run_id": run.id,
                                    "member_index": member_num,
                                    "member_name": f"{spec.model_id}_member_{member_num}",
                                },
                            )
                            _get_or_create(
                                db,
                                EnsembleMemberProductRecord,
                                (EnsembleMemberProductRecord.run_id == run.id)
                                & (EnsembleMemberProductRecord.member_index == member_num)
                                & (EnsembleMemberProductRecord.lead_time_hours == lead_time_hours),
                                {
                                    "id": f"member_product_{member_num}_{lead_time_hours}_{run.id}",
                                    "run_id": run.id,
                                    "member_index": member_num,
                                    "lead_time_hours": lead_time_hours,
                                },
                            )

                if committed_for_lead and spec.variables:
                    grid_code = spec.grid_id
                    product_type = spec.product_type
                    zarr_chunk_path = spec.zarr_store_path or run.zarr_store_path
                    if is_postgres:
                        from sqlalchemy.dialects.postgresql import insert as pg_insert

                        prod_values = [
                            {
                                "id": (
                                    f"product_{run.id}_{v.code}_{grid_code}_"
                                    f"{product_type}_{lead_time_hours}"
                                ),
                                "run_id": run.id,
                                "variable_id": v.code,
                                "grid_id": grid_code,
                                "product_type": product_type,
                                "lead_time_hours": lead_time_hours,
                                "zarr_chunk_path": zarr_chunk_path,
                            }
                            for v in spec.variables
                        ]
                        stmt_prod = pg_insert(ProductRecord).values(prod_values)
                        stmt_prod = stmt_prod.on_conflict_do_nothing(
                            index_elements=[
                                ProductRecord.run_id,
                                ProductRecord.variable_id,
                                ProductRecord.grid_id,
                                ProductRecord.product_type,
                                ProductRecord.lead_time_hours,
                            ]
                        )
                        db.execute(stmt_prod)
                    else:
                        for v in spec.variables:
                            _get_or_create(
                                db,
                                ProductRecord,
                                (ProductRecord.run_id == run.id)
                                & (ProductRecord.variable_id == v.code)
                                & (ProductRecord.grid_id == grid_code)
                                & (ProductRecord.product_type == product_type)
                                & (ProductRecord.lead_time_hours == lead_time_hours),
                                {
                                    "id": (
                                        f"product_{run.id}_{v.code}_{grid_code}_"
                                        f"{product_type}_{lead_time_hours}"
                                    ),
                                    "run_id": run.id,
                                    "variable_id": v.code,
                                    "grid_id": grid_code,
                                    "product_type": product_type,
                                    "lead_time_hours": lead_time_hours,
                                    "zarr_chunk_path": zarr_chunk_path,
                                },
                            )
                db.commit()

            # Advance manifest generation with new serving fingerprint
            run_identity = {
                "model_version_id": spec.version_string,
                "cycle_time": spec.cycle_time.isoformat(),
                "is_ensemble": spec.is_ensemble,
            }
            store_schema_fp = _store_schema_fingerprint(self.store_path, snapshot=self._snapshot)
            generation = _new_generation()
            manifest_payload = {
                "manifest_schema_version": 1,
                "store_protocol_mode": mode,
                "storage_format_version": getattr(settings, "STORAGE_FORMAT_VERSION", "sharded_v1"),
                "generation": generation,
                "run_identity": run_identity,
                "canonical_store_identity_hash": _store_identity_hash(self.store_path),
                "serving_state_fingerprint": sha256_hex("lead_pub", str(lead_time_hours), generation),
                "committed_state_fingerprint": sha256_hex(
                    "lead_committed", str(lead_time_hours), str(sorted(committed_for_lead))
                ),
                "store_schema_fingerprint": store_schema_fp,
                "region_marker_set_fingerprint": "",
                "legacy_region_evidence_fingerprint": None,
            }
            write_manifest(self.store_path, manifest_payload)
        finally:
            co.release_exclusive_gate()
            co.release_admission()

    def _marker_evidence_valid(
        self,
        region_id: str,
        payload: Mapping[str, object],
        *,
        existing_objects: set[str] | None = None,
        verify_physical_objects: bool = False,
    ) -> bool:
        """Validate a COMPLETE marker's completion evidence structurally.

        Returns True when the marker is internally consistent and matches chunk
        geometry and write-set fingerprint. When verify_physical_objects is True or
        existing_objects is provided, physical storage existence is checked as well.

        Args:
            region_id: The logical region id (``det_L0006`` / ``mem017_L0006``).
            payload: The marker body.
            existing_objects: Optional set of existing physical chunk keys.
            verify_physical_objects: Whether to check physical storage existence.

        Returns:
            True when the evidence is valid; False when the region is
            uncommitted.
        """
        from ingestion.core.inventory import (
            InventoryError,
            region_expected_object_keys,
            validate_marker_evidence,
        )

        try:
            member, lead = _parse_region_id(region_id)
            required_raw = payload.get("required_materialized_object_keys")
            omitted_raw = payload.get("intentionally_omitted_fill_chunks")
            fingerprint = payload.get("expected_write_set_fingerprint")
            if not isinstance(required_raw, list) or not isinstance(omitted_raw, list):
                return False
            if not isinstance(fingerprint, str):
                return False
            required = [str(k) for k in required_raw]
            omitted = [str(k) for k in omitted_raw]
            # Derive the actual expected write set from the cached .zarray
            # geometry + cached lead index (avoids per-marker store opens).
            lead_index = self._lead_index_for(lead)
            data_var_paths = sorted({k.split("/")[0] for k in required + omitted})
            if not data_var_paths:
                data_var_paths = _store_data_var_paths(self.store_path, snapshot=self._snapshot)
            is_sharded = any(k.endswith(".shard") for k in required + omitted)
            zarray_cache = (
                self._snapshot.zarray_by_var
                if self._snapshot is not None and self._snapshot.store_path == self.store_path
                else self._zarray_cache
            )
            zattrs_cache = (
                self._snapshot.zattrs_by_var
                if self._snapshot is not None and self._snapshot.store_path == self.store_path
                else self._zattrs_cache
            )
            member_index_cache = (
                self._snapshot.member_index_map
                if self._snapshot is not None and self._snapshot.store_path == self.store_path
                else self._member_index_cache
            )
            expected_keys = region_expected_object_keys(
                self.store_path,
                member=member,
                lead_index=lead_index,
                lead_time_hours=lead,
                format_version="sharded_v1" if is_sharded else "v2_unsharded",
                data_var_paths=data_var_paths,
                zarray_cache=zarray_cache,
                zattrs_cache=zattrs_cache,
                member_index_cache=member_index_cache,
            )
            validate_marker_evidence(
                self.store_path,
                marker_required_materialized=required,
                marker_omitted=omitted,
                actual_expected_keys=expected_keys,
                marker_expected_fingerprint=fingerprint,
                existing_objects=existing_objects,
                verify_physical_objects=verify_physical_objects,
            )
            return True
        except (InventoryError, ValueError, KeyError):
            logger.warning(
                "finalizer rejected COMPLETE marker evidence for region %s; "
                "treating it as uncommitted",
                region_id,
            )
            return False

    def _committed_state_from_regions(
        self,
        committed: dict[str, str],
        updating: list[str],
        expected_leads: tuple[int, ...],
        expected_members: tuple[int, ...],
        mode: str,
    ) -> CommittedState:
        # Convert region ids back to (lead, member). If any target is UPDATING,
        # the run cannot be ready (the finalizer's status derivation keeps it
        # partial when an expected region is not committed).
        leads: set[int] = set()
        pairs: set[tuple[int, int]] = set()
        for region_id in committed:
            member, lead = _parse_region_id(region_id)
            leads.add(lead)
            if member is not None:
                pairs.add((member, lead))
        # The store's real variable set (used for catalog ↔ store variable
        # honesty during reconciliation). Read once from the store's schema.
        store_vars = set(_store_data_var_paths(self.store_path, snapshot=self._snapshot))
        if self.spec.is_ensemble:
            members = {m for m, _ in pairs}
            return CommittedState.ensemble(pairs, members, variables=store_vars)
        return CommittedState.deterministic(leads, variables=store_vars)


def _validate_store_identity(
    dataset: xr.Dataset, existing: xr.Dataset, store_path: str
) -> None:
    from ingestion.core.pipeline import _validate_store_identity as vi

    vi(dataset, existing, store_path)


def _validate_lead_schema(
    dataset: xr.Dataset, existing: xr.Dataset, store_path: str
) -> None:
    from ingestion.core.pipeline import _validate_lead_schema as vl

    vl(dataset, existing, store_path)


def _read_marker_payload(store_path: str, key: str) -> dict[str, object]:
    """Read a marker by its key using the public marker API.

    The key is ``__commit__/v1/regions/<region>.json``; the region id is the
    basename without the ``.json`` suffix.
    """
    from ingestion.core.markers import read_region_marker

    region_id = key.rsplit("/", 1)[-1].removesuffix(".json")
    if region_id.startswith("det_"):
        lead = int(region_id[len("det_L") :])
        return read_region_marker(store_path, lead_time_hours=lead, member=None)
    if region_id.startswith("mem"):
        _, _, rest = region_id.partition("_L")
        member = int(region_id[3:6])
        lead = int(rest)
        return read_region_marker(store_path, lead_time_hours=lead, member=member)
    return {"state": "absent"}


def _read_marker_payloads_bounded(
    store_path: str,
    marker_keys: list[str],
    *,
    max_concurrency: int | None = None,
) -> list[tuple[str, dict[str, object]]]:
    """Read marker payloads for the given keys using bounded thread concurrency.

    Preserves input key order in the returned tuples. If any marker read fails
    with an exception, the exception is propagated immediately and all worker
    resources are cleaned up.

    Args:
        store_path: Root of the Zarr store.
        marker_keys: List of marker relative keys to read.
        max_concurrency: Maximum number of concurrent worker threads. Defaults
            to ``settings.MARKER_GET_CONCURRENCY`` (clamped to at least 1).

    Returns:
        List of (key, payload) tuples in the exact order of ``marker_keys``.
    """
    if not marker_keys:
        return []

    from ingestion.core.config import settings

    concurrency = int(
        max_concurrency
        if max_concurrency is not None
        else getattr(settings, "MARKER_GET_CONCURRENCY", 32)
    )
    concurrency = max(1, min(concurrency, len(marker_keys)))

    if concurrency == 1 or len(marker_keys) == 1:
        return [(k, _read_marker_payload(store_path, k)) for k in marker_keys]

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_read_marker_payload, store_path, k) for k in marker_keys
        ]
        try:
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
        except BaseException:
            for fut in futures:
                fut.cancel()
            raise

        payloads = [fut.result() for fut in futures]
        return list(zip(marker_keys, payloads, strict=True))


def _load_lead_indices_in_store(store_path: str) -> dict[int, int]:
    """Load all positional lead indices from the store's coordinate axis in one read."""
    from ingestion.core.zarr_writer import _resolve_store
    import zarr  # type: ignore[import-untyped]

    resolved = _resolve_store(store_path)
    try:
        ds = xr.open_zarr(resolved, consolidated=False)
        values = ds.coords["lead_time_hours"].values
        ds.close()
    except Exception:
        root = zarr.open_group(resolved, mode="r")
        values = root["lead_time_hours"][:]
    flat = np.atleast_1d(values).reshape(-1)
    return {int(v): i for i, v in enumerate(flat)}


def _lead_index_in_store(store_path: str, lead_time_hours: int) -> int:
    """Return the positional index of a lead in the store's coordinate axis."""
    indices = _load_lead_indices_in_store(store_path)
    if lead_time_hours not in indices:
        raise ValueError(
            f"lead {lead_time_hours} not found in store coordinate "
            f"lead_time_hours: {sorted(indices.keys())}"
        )
    return indices[lead_time_hours]


def _write_set_fingerprint(dataset: xr.Dataset) -> str:
    return sha256_hex("write-set", *sorted(str(v) for v in dataset.data_vars))


def _materialized_keys(dataset: xr.Dataset) -> list[str]:
    # The required materialized object set for a fresh real-data write is every
    # expected physical chunk of the region (no fill chunks are omitted on a
    # fresh write with real data; write_empty_chunks=False means a fully-fill
    # chunk is absent, but a real-data write materializes all its chunks).
    return sorted(f"{name}/" for name in dataset.data_vars)


def _omitted_fill_chunks(dataset: xr.Dataset) -> list[str]:
    # No fill chunks are omitted on a fresh write with real data; the empty
    # set is the honest attestation.
    return []


def _store_data_var_paths(
    store_path: str, snapshot: StoreMetadataSnapshot | None = None
) -> list[str]:
    """Return the data-variable array paths present in the store."""
    if snapshot is not None and snapshot.store_path == store_path:
        return list(snapshot.data_var_paths)
    try:
        from ingestion.core.zarr_writer import _resolve_store
        import zarr

        resolved = _resolve_store(store_path)
        root = zarr.open_group(resolved, mode="r")
        non_data = {"lead_time_hours", "latitude", "longitude", "member", "time"}
        return sorted(str(k) for k in root.keys() if str(k) not in non_data)
    except Exception:  # noqa: BLE001 - unreadable store -> empty
        return []


def _store_schema_fingerprint(
    store_path: str, snapshot: StoreMetadataSnapshot | None = None
) -> str:
    if snapshot is not None and snapshot.store_path == store_path:
        coords = sorted(
            list(snapshot.coords_values.keys())
            + (["member"] if snapshot.is_ensemble else [])
            + ["lead_time_hours"]
        )
        return sha256_hex(
            "schema",
            *coords,
            *sorted(snapshot.data_var_paths),
        )
    try:
        from ingestion.core.zarr_writer import _resolve_store
        import zarr

        resolved = _resolve_store(store_path)
        root = zarr.open_group(resolved, mode="r")
        coord_keys = {"lead_time_hours", "latitude", "longitude", "member", "time"}
        coords = sorted(str(k) for k in root.keys() if str(k) in coord_keys)
        vars = sorted(str(k) for k in root.keys() if str(k) not in coord_keys)
        return sha256_hex(
            "schema",
            *coords,
            *vars,
        )
    except Exception:  # noqa: BLE001
        return sha256_hex("schema-unreadable")


def _store_identity_hash(store_path: str) -> str:
    from domain.locks import canonical_storage_identity

    return sha256_hex(canonical_storage_identity(store_path))


def _region_serving_states(
    committed: dict[str, str],
    updating: list[str],
    mode: str,
) -> list[Mapping[str, object]]:
    out: list[Mapping[str, object]] = []
    for region_id in sorted(committed):
        member, lead = _parse_region_id(region_id)
        out.append(
            {
                "region": region_id,
                "state": "complete",
                "generation": committed[region_id],
                "member": member if member is not None else "",
                "lead": lead,
            }
        )
    for region_id in sorted(updating):
        out.append(
            {
                "region": region_id,
                "state": "updating",
                "generation": "",
                "member": "",
                "lead": "",
            }
        )
    return out


def _parse_region_id(region_id: str) -> tuple[int | None, int]:
    """Parse a logical region id (``det_L0006`` / ``mem017_L0006``)."""
    if region_id.startswith("det_"):
        lead = int(region_id[len("det_L") :])
        return None, lead
    if region_id.startswith("mem"):
        _, _, rest = region_id.partition("_L")
        member = int(region_id[3:6])
        lead = int(rest)
        return member, lead
    raise ValueError(f"cannot parse region id {region_id!r}")
