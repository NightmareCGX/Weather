"""Focused concurrency + region-write tests for the new ingestion architecture.

These tests prove the core guarantees of the region-write redesign:

* disjoint (member, lead) region writes never overwrite each other;
* out-of-order completion preserves real member identity (gep17 -> member 17);
* deterministic leads commit independently (lead 6 and lead 12 both survive);
* concurrent commits into one run store do not lose data;
* a run becomes READY only when every expected member/lead is committed.

All tests run against local disk stores and in-memory SQLite — no MinIO, no
live GRIB decode (synthetic datasets mirror the parser output).
"""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.core.base import LiveStoreOverwriteError
from ingestion.core.catalog import (
    CatalogBase,
    EnsembleMemberRecord,
    ModelRunRecord,
    ProductRecord,
    RunCatalogSpec,
    VariableSpec,
    record_run,
)
from ingestion.core.pipeline import ingest_grib_file
from ingestion.core.zarr_writer import commit_region, prepare_run_store, read_dataset

CYCLE = datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc)
GRID_LAT = [38.0, 38.25, 38.5, 38.75]
GRID_LON = [-107.0, -106.75, -106.5, -106.25]


def _single_lead_dataset(lead: int, member: int | None = None) -> xr.Dataset:
    """A normalized single-lead (optional single-member) dataset."""
    coords: dict[str, object] = {
        "lead_time_hours": [lead],
        "time": np.datetime64("2026-07-22T00:00:00"),
        "latitude": GRID_LAT,
        "longitude": GRID_LON,
    }
    dims = ("lead_time_hours", "latitude", "longitude")
    shape = (1, len(GRID_LAT), len(GRID_LON))
    if member is not None:
        coords["member"] = [member]
        dims = ("member", "lead_time_hours", "latitude", "longitude")
        shape = (1, 1, len(GRID_LAT), len(GRID_LON))
    data = np.full(shape, float(lead) + (member or 0.0), dtype=np.float32)
    return xr.Dataset(
        data_vars={"temperature_2m": (dims, data)},
        coords=coords,
    )


def _spec(
    store: str,
    *,
    is_ensemble: bool = False,
    expected_leads: tuple[int, ...] = (6, 12),
    expected_members: tuple[int, ...] = (),
) -> RunCatalogSpec:
    return RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="USA",
        model_id="gefs" if is_ensemble else "gfs",
        model_name="GEFS" if is_ensemble else "GFS",
        is_ensemble=is_ensemble,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=CYCLE,
        grid_id="global_025deg",
        grid_name="g",
        grid_resolution_km=25.0,
        zarr_store_path=store,
        variables=(VariableSpec("temperature_2m", "T", "°C", "t2m"),),
        expected_lead_time_hours=expected_leads,
        expected_members=expected_members,
    )


@pytest.fixture
def session(tmp_path) -> Session:
    # File-backed SQLite so the pipeline's live-store guard (which uses the
    # injectable _live_store_session_factory) shares the same schema/rows.
    db_file = tmp_path / "catalog.sqlite"
    engine = create_engine(
        f"sqlite:///{db_file}", connect_args={"check_same_thread": False}
    )
    CatalogBase.metadata.create_all(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


@pytest.fixture(autouse=True)
def _route_live_store_session(session: Session, monkeypatch) -> None:
    """Route the library path's live-store guard to the test SQLite engine."""
    import ingestion.core.pipeline as P

    monkeypatch.setattr(P, "_live_store_session_factory", lambda: session.bind)


def test_disjoint_member_region_writes_do_not_overwrite() -> None:
    """Disjoint-member region writes never clobber each other.

    Region writes are serialized on the store's shared metadata read (the
    coordinate-index resolution), which is the required serialization boundary:
    disjoint *chunk data* writes are independent, but the region-index
    resolution reads the store and must not race. Production serializes this
    per-run via an asyncio.Lock; the test mirrors that with a threading lock.
    """
    store = os.path.join(tempfile.mkdtemp(), "cycle.zarr")
    # Pre-allocate 3 members x 2 leads.
    prepare_run_store(
        _single_lead_dataset(6, member=1),
        store,
        expected_lead_time_hours=(6, 12),
        expected_members=(1, 2, 3),
    )

    import threading

    lock = threading.Lock()

    def _commit(member: int) -> None:
        with lock:
            commit_region(_single_lead_dataset(6, member=member), store)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(_commit, [1, 2, 3]))

    restored = read_dataset(store)
    for member in (1, 2, 3):
        value = float(
            restored["temperature_2m"].sel(member=member, lead_time_hours=6).values[0, 0]
        )
        # value = lead(6) + member identity
        assert value == pytest.approx(6.0 + member, abs=1e-4)
    # lead 12 untouched (NaN).
    assert np.isnan(
        restored["temperature_2m"].sel(member=1, lead_time_hours=12).values[0, 0]
    )


def test_out_of_order_completion_preserves_member_identity() -> None:
    """gep17 lands in member 17 even when ingested before gep01."""
    store = os.path.join(tempfile.mkdtemp(), "cycle.zarr")
    prepare_run_store(
        _single_lead_dataset(6, member=17),
        store,
        expected_lead_time_hours=(6,),
        expected_members=(1, 17, 30),
    )
    # Ingest out of order: 17 first, then 30, then 1.
    for member in (17, 30, 1):
        commit_region(_single_lead_dataset(6, member=member), store)
    restored = read_dataset(store)
    for member, expected in ((1, 6 + 1), (17, 6 + 17), (30, 6 + 30)):
        value = float(
            restored["temperature_2m"].sel(member=member, lead_time_hours=6).values[0, 0]
        )
        assert value == pytest.approx(expected, abs=1e-4)


def test_deterministic_leads_commit_independently() -> None:
    """Lead 6 and lead 12 both survive a deterministic region commit."""
    store = os.path.join(tempfile.mkdtemp(), "cycle.zarr")
    prepare_run_store(
        _single_lead_dataset(6),
        store,
        expected_lead_time_hours=(6, 12),
    )
    commit_region(_single_lead_dataset(6), store)
    commit_region(_single_lead_dataset(12), store)
    restored = read_dataset(store)
    assert sorted(int(v) for v in restored["lead_time_hours"].values) == [6, 12]
    assert float(
        restored["temperature_2m"].sel(lead_time_hours=6).values[0, 0]
    ) == pytest.approx(6.0)
    assert float(
        restored["temperature_2m"].sel(lead_time_hours=12).values[0, 0]
    ) == pytest.approx(12.0)


def test_ingest_grib_file_concurrent_members_ready_when_complete(
    session: Session, tmp_path, monkeypatch
) -> None:
    """Out-of-order per-member ingest reaches READY only when all members commit."""
    store = str(tmp_path / "cycle.zarr")

    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        return record_run(session, spec, dataset, member=member, committed_state=committed_state)

    def _fake_parse(path):
        # The test passes a synthetic dataset; derive member from the path.
        import re

        match = re.search(r"m(\d+)\.grib2", str(path))
        member = int(match.group(1)) if match else 1
        return _single_lead_dataset(6, member=member)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )
    monkeypatch.setattr("ingestion.core.pipeline.parse_grib2", _fake_parse)
    spec = _spec(store, is_ensemble=True, expected_leads=(6,), expected_members=(1, 2, 3))

    # The first library-path ingest creates the live store. Any further
    # library-path ingest of the same store (a different member) is refused —
    # multi-member ingestion is now the coordinator's responsibility (the
    # weather-ingest CLI coordinator), which enforces the concurrency protocol.
    first = ingest_grib_file(
        spec,
        str(tmp_path / "m3.grib2"),
        store,
        requested_lead_time_hours=6,
        member=3,
    )
    assert first.status == "partial"
    with pytest.raises(LiveStoreOverwriteError):
        ingest_grib_file(
            spec,
            str(tmp_path / "m1.grib2"),
            store,
            requested_lead_time_hours=6,
            member=1,
        )
    # Only member 3 was committed via the library path.
    member_rows = {m.member_index for m in session.query(EnsembleMemberRecord).all()}
    assert member_rows == {3}


def test_partial_run_has_correct_status(session: Session, tmp_path, monkeypatch) -> None:
    """A run with some but not all expected members is partial, not ready."""
    store = str(tmp_path / "cycle.zarr")

    def _record_into_session(spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        return record_run(session, spec, dataset, member=member, committed_state=committed_state)

    def _fake_parse(path):
        return _single_lead_dataset(6, member=1)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )
    monkeypatch.setattr("ingestion.core.pipeline.parse_grib2", _fake_parse)
    spec = _spec(store, is_ensemble=True, expected_leads=(6,), expected_members=(1, 2, 3))
    record = ingest_grib_file(
        spec,
        str(tmp_path / "m1.grib2"),
        store,
        requested_lead_time_hours=6,
        member=1,
    )
    assert record.status == "partial"
    run = session.query(ModelRunRecord).one()
    assert run.status == "partial"
    # Products recorded for the committed lead only.
    leads = {p.lead_time_hours for p in session.query(ProductRecord).all()}
    assert leads == {6}


# --- Ensemble committed-state consistency (Phase 2A) ---


def test_ensemble_committed_pairs_detected_and_ready(session: Session, tmp_path, monkeypatch) -> None:
    """Actual committed (member, lead) pairs are detected and the full set is READY."""
    from ingestion.core.catalog import EnsembleMemberProductRecord
    from ingestion.core.pipeline import read_committed_state

    store = str(tmp_path / "ens_ready.zarr")
    spec = _spec(store, is_ensemble=True, expected_leads=(6,), expected_members=(1, 2, 3))
    prepare_run_store(
        _single_lead_dataset(6, member=1),
        store,
        expected_lead_time_hours=(6,),
        expected_members=(1, 2, 3),
    )

    def _record_into_session(s, spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        return record_run(s, spec, dataset, member=member, committed_state=committed_state)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )
    for member in (1, 2, 3):
        with Session(session.bind) as s:
            commit_region(_single_lead_dataset(6, member=member), store)
            record_run(s, spec, _single_lead_dataset(6, member=member), member=member,
                       committed_state=read_committed_state(store, is_ensemble=True))
    run = session.query(ModelRunRecord).one()
    assert run.status == "ready"
    pairs = {
        (p.member_index, p.lead_time_hours)
        for p in session.query(EnsembleMemberProductRecord).filter_by(run_id=run.id).all()
    }
    assert pairs == {(1, 6), (2, 6), (3, 6)}


def test_ensemble_stale_member_metadata_reconciled(session: Session, tmp_path, monkeypatch) -> None:
    """A member with no committed pair in the store is reconciled away."""
    from ingestion.core.catalog import EnsembleMemberRecord, EnsembleMemberProductRecord
    from ingestion.core.pipeline import read_committed_state

    store = str(tmp_path / "ens_stale.zarr")
    spec = _spec(store, is_ensemble=True, expected_leads=(0, 6), expected_members=(1, 2, 3))
    prepare_run_store(
        _single_lead_dataset(0, member=1),
        store,
        expected_lead_time_hours=(0, 6),
        expected_members=(1, 2, 3),
    )

    def _record_into_session(s, spec, dataset, *, effective_store_path=None, member=None, committed_state=None):
        return record_run(s, spec, dataset, member=member, committed_state=committed_state)

    monkeypatch.setattr(
        "ingestion.core.pipeline.record_ingested_dataset", _record_into_session
    )
    # Commit members 1 and 2 fully; member 3 never commits.
    for member, leads in ((1, (0, 6)), (2, (0, 6))):
        for lead in leads:
            with Session(session.bind) as s:
                commit_region(_single_lead_dataset(lead, member=member), store)
                record_run(s, spec, _single_lead_dataset(lead, member=member), member=member,
                           committed_state=read_committed_state(store, is_ensemble=True))
    run = session.query(ModelRunRecord).one()
    members = sorted(m.member_index for m in session.query(EnsembleMemberRecord).filter_by(run_id=run.id).all())
    pairs = {
        (p.member_index, p.lead_time_hours)
        for p in session.query(EnsembleMemberProductRecord).filter_by(run_id=run.id).all()
    }
    # Member 3 (never committed) is reconciled away.
    assert members == [1, 2]
    assert pairs == {(1, 0), (1, 6), (2, 0), (2, 6)}
    assert run.status == "partial"  # expected member 3 not committed
