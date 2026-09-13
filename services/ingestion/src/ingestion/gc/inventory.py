"""Store <-> catalog reconciliation: orphan cycle inventory (Lifecycle V3 §9).

The system is monotonic: old cycles are never re-activated by the realtime
scheduler. A physical cycle store with no catalog identity is therefore an
orphan, and reconciliation follows a simple recoverability frontier:

    store exists + catalog missing
        -> inside the recoverable frontier?
            yes -> catalog recovery from COMPLETE marker evidence is possible
                   (deliberately NOT automated here; operator decision)
            no  -> beyond the frontier: catalog recovery will never happen;
                   the store is eligible for orphan cleanup

No anti-resurrection subsystem is required: the permanent lifecycle tombstone
is only consulted as a sanity guard during reaping. Orphan stores have no
catalog identity and therefore no catalog-tracked dependencies — every object
in the prefix is untracked by definition, which is what makes prefix cleanup
here consistent with the dependency-driven GC model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from domain.horizon import model_max_lead_hours
from domain.lifecycle import is_cycle_horizon_expired
from domain.temporal import model_serving_start_valid_time, serving_start_valid_time
from ingestion.core.catalog import ForecastCycleLifecycleRecord, ModelRunRecord, ModelVersionRecord
from ingestion.core.locks import LockTimeoutError, StoreLockCoordinator

logger = logging.getLogger(__name__)

#: Canonical cycle store layout: <root>/<model>/<YYYY-MM-DD>/<HH>/cycle.zarr
_CYCLE_STORE_RE = re.compile(
    r"^(?P<root>.*)/(?P<model>[^/]+)/(?P<date>\d{4}-\d{2}-\d{2})/(?P<hour>\d{2})/cycle\.zarr$"
)


@dataclass(frozen=True)
class OrphanStore:
    """A physical cycle store with no catalog identity."""

    model_id: str
    cycle_time: datetime
    store_path: str
    beyond_frontier: bool


@dataclass(frozen=True)
class InventoryResult:
    """Structured result of one orphan inventory pass."""

    evaluated_at: datetime
    cataloged_cycles: int
    discovered_stores: int
    orphan_stores: tuple[OrphanStore, ...]
    reaped_stores: tuple[str, ...]
    reap_blocked: tuple[str, ...]
    errors: tuple[str, ...]


def list_physical_cycle_stores(store_root: str) -> list[tuple[str, datetime, str]]:
    """Enumerate physical cycle stores under ``store_root``.

    Supports ``s3://`` roots (via the shared control-plane S3 filesystem) and
    local filesystem roots (tests / single-node deployments). Returns
    ``(model_id, cycle_time, store_path)`` triples for canonical-layout stores.
    Non-conforming paths are ignored.
    """
    results: list[tuple[str, datetime, str]] = []
    if store_root.startswith("s3://"):
        from ingestion.core.config import settings
        from ingestion.core.s3 import get_control_s3_fs

        fs = get_control_s3_fs(settings)
        pattern = store_root.rstrip("/") + "/*/*/*/cycle.zarr"
        candidates = fs.glob(pattern)
    else:
        root = Path(store_root)
        candidates = [str(p) for p in root.glob("*/*/*/cycle.zarr")]

    for raw in candidates:
        path = raw.replace("\\", "/")
        match = _CYCLE_STORE_RE.match(path)
        if match is None:
            continue
        model = match.group("model").lower().strip()
        try:
            cycle_time = datetime.fromisoformat(
                f"{match.group('date')}T{match.group('hour')}:00:00+00:00"
            )
        except ValueError:
            continue
        if store_root.startswith("s3://") and not path.startswith("s3://"):
            path = f"s3://{path}"
        results.append((model, cycle_time, path))
    return results


def discover_orphan_cycles(
    session: Session,
    *,
    store_cycles: list[tuple[str, datetime, str]],
    now: datetime,
) -> tuple[list[OrphanStore], int]:
    """Classify physical stores against the catalog identity set.

    A store is cataloged iff any model_runs row exists for its
    (model_id, cycle_time) — any version, any status. Otherwise it is an
    orphan; ``beyond_frontier`` marks orphans whose entire possible serving
    horizon is past (never re-activatable, catalog recovery pointless).
    """
    cataloged: set[tuple[str, datetime]] = {
        (str(m).lower().strip(), _ensure_utc(c))
        for m, c in session.execute(
            select(ModelVersionRecord.model_id, ModelRunRecord.cycle_time)
            .join(ModelVersionRecord, ModelRunRecord.model_version_id == ModelVersionRecord.id)
        ).all()
    }

    now_utc = _ensure_utc(now)
    serving_start = serving_start_valid_time(now_utc)
    orphans: list[OrphanStore] = []
    seen: set[tuple[str, datetime, str]] = set()
    for model, cycle_time, store_path in store_cycles:
        key = (model, cycle_time, store_path)
        if key in seen:
            continue
        seen.add(key)
        if (model, cycle_time) in cataloged:
            continue
        try:
            max_lead = model_max_lead_hours(model)
        except ValueError:
            max_lead = 240
        try:
            model_serving_start = model_serving_start_valid_time(model, now_utc)
        except ValueError:
            # Unregistered model: fall back to the canonical global boundary
            # (same tolerance as model_max_lead_hours above).
            model_serving_start = serving_start
        beyond = is_cycle_horizon_expired(
            cycle_time, max_lead_hours=max_lead, serving_start=model_serving_start
        )
        orphans.append(
            OrphanStore(
                model_id=model,
                cycle_time=cycle_time,
                store_path=store_path,
                beyond_frontier=beyond,
            )
        )
    cataloged_count = len(cataloged)
    return orphans, cataloged_count


def reap_orphan_store(
    engine: Engine,
    orphan: OrphanStore,
    *,
    timeout_seconds: float = 5.0,
) -> bool:
    """Delete one orphan store prefix under an EXCLUSIVE store gate.

    Sanity guards (fail closed):
    - a lifecycle row exists for the cycle (tombstone guard — a tombstoned
      cycle is never an orphan candidate);
    - a model_runs row appeared since discovery (raced catalog recovery).
    """
    with Session(engine) as session:
        lc = session.get(ForecastCycleLifecycleRecord, (orphan.model_id, orphan.cycle_time))
        if lc is not None:
            logger.warning(
                "orphan_reap_refused_lifecycle_row: model=%s cycle=%s",
                orphan.model_id,
                orphan.cycle_time.isoformat(),
            )
            return False
        raced = session.execute(
            select(ModelRunRecord.id)
            .join(ModelVersionRecord, ModelRunRecord.model_version_id == ModelVersionRecord.id)
            .where(
                ModelVersionRecord.model_id == orphan.model_id,
                ModelRunRecord.cycle_time == orphan.cycle_time,
            )
            .limit(1)
        ).scalar_one_or_none()
        if raced is not None:
            logger.warning(
                "orphan_reap_refused_raced_catalog: model=%s cycle=%s",
                orphan.model_id,
                orphan.cycle_time.isoformat(),
            )
            return False

    if orphan.store_path.startswith("s3://"):
        from ingestion.core.config import settings
        from ingestion.core.s3 import get_control_s3_fs

        fs = get_control_s3_fs(settings)
        raw_path = orphan.store_path[len("s3://") :].rstrip("/")

        def _rm() -> None:
            try:
                if fs.exists(raw_path):
                    fs.rm(raw_path, recursive=True)
            except Exception as exc:  # noqa: BLE001 - missing prefix is success
                logger.debug("orphan_reap_s3_rm_error on %s: %s", raw_path, exc)

    else:
        local = Path(orphan.store_path)

        def _rm() -> None:
            import shutil

            if local.is_dir():
                shutil.rmtree(local, ignore_errors=True)
            elif local.exists():
                local.unlink(missing_ok=True)

    if engine.dialect.name != "postgresql":
        _rm()
        return True

    with engine.connect() as conn:
        coord = StoreLockCoordinator(
            conn, store_path=orphan.store_path, timeout_seconds=timeout_seconds
        )
        try:
            coord.acquire_exclusive_gate()
        except LockTimeoutError:
            logger.warning(
                "orphan_reap_gate_blocked: store_path=%s", orphan.store_path
            )
            return False
        try:
            _rm()
        finally:
            coord.release_exclusive_gate()
    return True


def run_orphan_inventory(
    engine: Engine,
    *,
    store_root: str,
    reap: bool = False,
    timeout_seconds: float = 5.0,
    now: datetime | None = None,
    catalog_session: Any = None,
) -> InventoryResult:
    """Run one orphan inventory pass (and optionally reap beyond-frontier orphans)."""
    from ingestion.core.catalog import _utcnow

    now_utc = _ensure_utc(now) if now is not None else _utcnow()
    store_cycles = list_physical_cycle_stores(store_root)

    own_session = catalog_session is None
    if own_session:
        with Session(engine) as session:
            orphans, cataloged_count = discover_orphan_cycles(
                session, store_cycles=store_cycles, now=now_utc
            )
    else:
        orphans, cataloged_count = discover_orphan_cycles(
            catalog_session, store_cycles=store_cycles, now=now_utc
        )

    reaped: list[str] = []
    blocked: list[str] = []
    if reap:
        for orphan in orphans:
            if not orphan.beyond_frontier:
                continue
            try:
                if reap_orphan_store(engine, orphan, timeout_seconds=timeout_seconds):
                    reaped.append(orphan.store_path)
                else:
                    blocked.append(orphan.store_path)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "orphan_reap_failed: store_path=%s error=%s",
                    orphan.store_path,
                    exc,
                )
                blocked.append(orphan.store_path)

    return InventoryResult(
        evaluated_at=now_utc,
        cataloged_cycles=cataloged_count,
        discovered_stores=len({(m, c) for m, c, _ in store_cycles}),
        orphan_stores=tuple(orphans),
        reaped_stores=tuple(reaped),
        reap_blocked=tuple(blocked),
        errors=(),
    )


def _ensure_utc(dt: datetime) -> datetime:
    from domain.lifecycle import _ensure_utc as _ensure

    return _ensure(dt)
