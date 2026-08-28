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
from ingestion.core.base import (
    StoreSchemaMismatchError,
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
    read_dataset,
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
        # Cache the data-variable .zarray geometry per array path.
        self._zarray_cache: dict[str, dict[str, object]] = {}

    def _lead_index_for(self, lead: int) -> int:
        """Return the positional lead index, caching per lead value."""
        if lead not in self._lead_index_cache:
            self._lead_index_cache[lead] = _lead_index_in_store(self.store_path, lead)
        return self._lead_index_cache[lead]

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
                existing = read_dataset(self.store_path)
                _validate_store_identity(seed_dataset, existing, self.store_path)
                if run_id is not None and is_same_cycle:
                    with Session(bind=conn) as db:
                        set_run_partial(db, run_id)
                        db.commit()
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
            region_ids = _physical_conflict_region_ids(
                dataset, self.store_path, member=member
            )
            co.acquire_region_locks(region_ids)
            try:
                lead_values = dataset.coords["lead_time_hours"].values
                lead = int(np.asarray(lead_values).reshape(-1)[0])
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
                # Generation-ownership confirmed: write the data.
                _commit_region(
                    dataset,
                    self.store_path,
                    member=member,
                    expected_lead_time_hours=expected_leads,
                    expected_members=expected_members,
                )
                # Compute the physical object inventory for the COMPLETE marker.
                from ingestion.core.inventory import (
                    expected_write_set_fingerprint,
                    list_object_keys,
                    region_expected_object_keys,
                )

                lead_index = _lead_index_in_store(self.store_path, lead)
                data_var_paths = sorted(str(v) for v in dataset.data_vars)
                expected_keys = region_expected_object_keys(
                    self.store_path,
                    member=member,
                    lead_index=lead_index,
                    data_var_paths=data_var_paths,
                )
                existing_keys = set()
                for var in data_var_paths:
                    existing_keys.update(list_object_keys(self.store_path, var))
                # Real-data writes materialize all expected chunks; an expected
                # chunk that is absent is an all-fill omission.
                required = [k for k in expected_keys if k in existing_keys]
                omitted = [k for k in expected_keys if k not in existing_keys]
                # Write COMPLETE marker (last store-side operation).
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
    ) -> FinalizeResult:
        """Run the single coalesced finalization for the bounded CLI wave.

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
            # Load marker states (LIST + GET, all pages).
            marker_keys = list_region_marker_keys(self.store_path)
            # Build the physical object inventory ONCE (one paginated LIST per
            # data-variable prefix) so per-marker existence checks are in-memory
            # instead of one remote exists per object.
            from ingestion.core.inventory import build_object_inventory

            data_var_paths = _store_data_var_paths(self.store_path)
            existing_objects = (
                build_object_inventory(self.store_path, data_var_paths)
                if data_var_paths
                else set()
            )
            # Populate the .zarray geometry cache once (per array path) so
            # per-marker region-key derivation avoids repeated remote reads.
            from ingestion.core.inventory import _read_zarray

            for array_path in data_var_paths:
                if array_path not in self._zarray_cache:
                    za = _read_zarray(self.store_path, array_path)
                    if za is not None:
                        self._zarray_cache[array_path] = za
            committed: dict[str, str] = {}  # region_id -> generation
            updating: list[str] = []
            for key in marker_keys:
                region_id = key.rsplit("/", 1)[-1].removesuffix(".json")
                payload = _read_marker_payload(self.store_path, key)
                state = payload.get("state")
                gen = payload.get("generation")
                if state == "complete":
                    # Full object-inventory validation of the COMPLETE marker's
                    # completion evidence. A structurally-invalid or externally-
                    # shrunk marker is treated as uncommitted (not cataloged).
                    if self._marker_evidence_valid(
                        region_id,
                        payload,
                        existing_objects=existing_objects,
                    ):
                        committed[region_id] = str(gen)
                    else:
                        updating.append(region_id)
                elif state == "updating":
                    updating.append(region_id)
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
            store_schema_fp = _store_schema_fingerprint(self.store_path)
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
                "generation": generation,
                "run_identity": run_identity,
                "canonical_store_identity_hash": _store_identity_hash(self.store_path),
                "serving_state_fingerprint": serving_fp,
                "committed_state_fingerprint": committed_fp,
                "store_schema_fingerprint": store_schema_fp,
                "region_marker_set_fingerprint": markers_fp,
                "legacy_region_evidence_fingerprint": legacy_fp,
            }
            write_manifest(self.store_path, payload)

            # Reconcile catalog to the committed state and derive status.
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
                observer.record_milestone("finalize_complete")
            return FinalizeResult(status=status, committed_regions=committed)
        finally:
            co.release_exclusive_gate()
            co.release_admission()

    def _marker_evidence_valid(
        self,
        region_id: str,
        payload: Mapping[str, object],
        *,
        existing_objects: set[str] | None = None,
    ) -> bool:
        """Validate a COMPLETE marker's completion evidence structurally.

        Returns True when the marker is internally consistent AND every required
        materialized object currently exists. A structurally-invalid or
        externally-shrunk marker is treated as uncommitted (not cataloged),
        preserving external-shrink detection.

        Args:
            region_id: The logical region id (``det_L0006`` / ``mem017_L0006``).
            payload: The marker body.

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
            expected_keys = region_expected_object_keys(
                self.store_path,
                member=member,
                lead_index=lead_index,
                data_var_paths=data_var_paths,
                zarray_cache=self._zarray_cache,
            )
            validate_marker_evidence(
                self.store_path,
                marker_required_materialized=required,
                marker_omitted=omitted,
                actual_expected_keys=expected_keys,
                marker_expected_fingerprint=fingerprint,
                existing_objects=existing_objects,
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
        store_vars = set(_store_data_var_paths(self.store_path))
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


def _lead_index_in_store(store_path: str, lead_time_hours: int) -> int:
    """Return the positional index of a lead in the store's coordinate axis."""
    ds = read_dataset(store_path)
    values = ds.coords["lead_time_hours"].values
    flat = np.atleast_1d(values).reshape(-1)
    for i, value in enumerate(flat):
        if int(value) == int(lead_time_hours):
            return i
    raise ValueError(
        f"lead {lead_time_hours} not found in store coordinate "
        f"lead_time_hours: {[int(v) for v in flat]}"
    )


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


def _store_data_var_paths(store_path: str) -> list[str]:
    """Return the data-variable array paths present in the store."""
    try:
        ds = read_dataset(store_path)
        return sorted(str(v) for v in ds.data_vars)
    except Exception:  # noqa: BLE001 - unreadable store -> empty
        return []


def _store_schema_fingerprint(store_path: str) -> str:
    try:
        ds = read_dataset(store_path)
        return sha256_hex(
            "schema",
            *sorted(str(c) for c in ds.coords),
            *sorted(str(v) for v in ds.data_vars),
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
