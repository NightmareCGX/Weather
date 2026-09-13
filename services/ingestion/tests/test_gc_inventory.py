"""Tests for the store-catalog orphan inventory (lifecycle-v3-architecture.md §9).

Covers:
* Physical store enumeration from a local root (canonical layout).
* Orphan classification: cataloged (any version/status) vs missing identity.
* Monotonic recoverability frontier: orphans whose whole possible serving
  horizon is past are beyond the frontier (reapable); otherwise they stay in
  the recoverable region (catalog recovery from marker evidence is possible).
* Reap sanity guards: refuse when a lifecycle row or a catalog run exists.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.core.catalog import (
    CatalogBase,
    CenterRecord,
    ForecastCycleLifecycleRecord,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
)
from ingestion.gc.inventory import (
    discover_orphan_cycles,
    list_physical_cycle_stores,
    reap_orphan_store,
)


def _dt(year: int, month: int, day: int, hour: int) -> datetime:
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def catalog_engine():
    engine = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                CenterRecord(id="c", center_id="noaa", name="NOAA", country="US"),
                ModelRecord(id="m", model_id="gfs", name="GFS", center_id="noaa", is_ensemble=False, resolution_km=25.0),
                ModelVersionRecord(id="v", model_id="gfs", version_string="v1.0"),
            ]
        )
        session.commit()
    yield engine
    engine.dispose()


def test_list_physical_cycle_stores_parses_canonical_layout(tmp_path):
    for model, date, hour in [
        ("gfs", "2026-09-12", "00"),
        ("gfs", "2026-09-12", "06"),
        ("gefs", "2026-09-12", "00"),
        ("gfs", "not-a-date", "00"),
    ]:
        d = tmp_path / model / date / hour
        d.mkdir(parents=True)
        (d / "cycle.zarr").mkdir()

    stores = list_physical_cycle_stores(str(tmp_path))
    parsed = {(m, c) for m, c, _ in stores}
    assert (_dt(2026, 9, 12, 0)) in {c for _, c in parsed}
    assert len(stores) == 3  # the non-conforming date is ignored


def test_orphan_classification_and_frontier(catalog_engine, tmp_path):
    # Cataloged cycle (any status) must NOT be an orphan even if unknown to us.
    with Session(catalog_engine) as session:
        session.add(
            ModelRunRecord(
                id="run_gfs_2026091200",
                model_version_id="v",
                cycle_time=_dt(2026, 9, 12, 0),
                status="partial",
                zarr_store_path=str(tmp_path / "gfs" / "2026-09-12" / "00" / "cycle.zarr"),
            )
        )
        session.commit()

    # Physical stores: one cataloged, one orphan inside the frontier, one beyond.
    for model, date, hour in [
        ("gfs", "2026-09-12", "00"),   # cataloged (partial run above)
        ("gfs", "2026-09-13", "00"),   # orphan, inside recoverable frontier
        ("gfs", "2026-01-01", "00"),   # orphan, far beyond frontier
    ]:
        d = tmp_path / model / date / hour
        d.mkdir(parents=True)
        (d / "cycle.zarr").mkdir()

    store_cycles = list_physical_cycle_stores(str(tmp_path))
    now = _dt(2026, 9, 13, 6)

    with Session(catalog_engine) as session:
        orphans, cataloged_count = discover_orphan_cycles(
            session, store_cycles=store_cycles, now=now
        )

    assert cataloged_count == 1
    by_path = {o.store_path: o for o in orphans}
    assert len(orphans) == 2
    inside = by_path[str(tmp_path / "gfs" / "2026-09-13" / "00" / "cycle.zarr").replace("\\", "/")]
    beyond = by_path[str(tmp_path / "gfs" / "2026-01-01" / "00" / "cycle.zarr").replace("\\", "/")]
    assert inside.beyond_frontier is False
    assert beyond.beyond_frontier is True


def test_reap_sanity_guards_fail_closed(catalog_engine, tmp_path):
    store = tmp_path / "gfs" / "2026-01-01" / "00" / "cycle.zarr"
    store.mkdir(parents=True)
    (store / "chunk.bin").write_text("x")
    orphan = type(
        "O",
        (),
        {
            "model_id": "gfs",
            "cycle_time": _dt(2026, 1, 1, 0),
            "store_path": str(store),
            "beyond_frontier": True,
        },
    )()

    # Guard 1: a lifecycle row (any state) refuses the reap.
    with Session(catalog_engine) as session:
        session.add(
            ForecastCycleLifecycleRecord(
                model_id="gfs",
                cycle_time=orphan.cycle_time,
                deletion_started_at=_dt(2026, 9, 13, 0),
            )
        )
        session.commit()
    assert reap_orphan_store(catalog_engine, orphan) is False
    assert store.exists()

    # Guard 2: a raced catalog run refuses the reap.
    with Session(catalog_engine) as session:
        session.delete(
            session.get(ForecastCycleLifecycleRecord, ("gfs", orphan.cycle_time))
        )
        session.add(
            ModelRunRecord(
                id="run_raced",
                model_version_id="v",
                cycle_time=orphan.cycle_time,
                status="ready",
                zarr_store_path=str(store),
            )
        )
        session.commit()
    assert reap_orphan_store(catalog_engine, orphan) is False
    assert store.exists()

    # Clean orphan: reap succeeds (SQLite harness: direct removal, no gate).
    with Session(catalog_engine) as session:
        session.delete(session.get(ModelRunRecord, "run_raced"))
        session.commit()
    assert reap_orphan_store(catalog_engine, orphan) is True
    assert not store.exists()
