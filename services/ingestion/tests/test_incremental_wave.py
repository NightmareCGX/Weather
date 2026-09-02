"""Phase 5B regression tests: repeated disjoint wave targets on one cycle.

These tests pin the wave-target / cycle-horizon split:

* Test A (disjoint subset accumulation) — multiple invocations with disjoint
  lead targets accumulate into ONE cycle store without ``StoreSchemaMismatchError``;
  the committed data and catalog contain the union; the run stays ``partial``
  because the canonical horizon is incomplete.
* Test B (eventual full-horizon convergence) — waves whose targets union to
  the full canonical horizon converge to ``ready`` (GEFS: the exact required
  ``member × lead`` pair set).

The tests exercise the real wave runner, coordinator, markers, sharded_v1
local stores, and catalog reconciliation. Upstream downloads and process-bound
decoding are stubbed (deterministic synthetic datasets per (member, lead));
the canonical horizon is reduced via the domain registry's test fixture
contract (``register_canonical_lead_horizon``) — production keeps the
canonical 81-lead horizon, which is asserted separately against ``_build_spec``.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.core.catalog import (
    CatalogBase,
    EnsembleMemberProductRecord,
    ModelRunRecord,
    ProductRecord,
)
from ingestion.core.wave_runner import RunSpec, _build_spec, _run_wave

HORIZON = tuple(range(0, 24, 3))  # reduced injected horizon: 8 leads
CYCLE = date(2026, 7, 21)


@pytest.fixture()
def reduced_horizon():
    """Inject a reduced canonical horizon (test fixture contract) and restore."""
    from domain.horizon import MODEL_CANONICAL_HORIZONS, register_canonical_lead_horizon

    saved = dict(MODEL_CANONICAL_HORIZONS)
    register_canonical_lead_horizon("gfs", HORIZON)
    register_canonical_lead_horizon("gefs", HORIZON)
    yield HORIZON
    MODEL_CANONICAL_HORIZONS.clear()
    MODEL_CANONICAL_HORIZONS.update(saved)


@pytest.fixture()
def catalog_db(tmp_path: Path):
    """File-based SQLite catalog engine (in-memory would not survive dispose)."""
    engine = create_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    CatalogBase.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def _noop_locks(monkeypatch: pytest.MonkeyPatch):
    """No-op the advisory-lock coordinator (SQLite has no PG advisory locks)."""
    monkeypatch.setattr(
        "ingestion.core.coordinator.StoreLockCoordinator", _NoopLockCoordinator
    )


class _NoopLockCoordinator:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def acquire_shared_gate(self) -> None:
        pass

    def release_shared_gate(self) -> None:
        pass

    def acquire_exclusive_gate(self) -> None:
        pass

    def release_exclusive_gate(self) -> None:
        pass

    def acquire_admission(self) -> None:
        pass

    def release_admission(self) -> None:
        pass

    def acquire_shared_admission(self) -> None:
        pass

    def release_shared_admission(self) -> None:
        pass

    def acquire_region_locks(self, region_ids: list[str]) -> None:
        pass

    def release_region_locks(self, region_ids: list[str]) -> None:
        pass

    def release_all(self) -> None:
        pass

    def close_connection(self) -> None:
        pass


def _synthetic_dataset(lead: int, member: int | None = None) -> xr.Dataset:
    """A deterministic single-lead (optionally single-member) parsed dataset.

    Uses platform variable codes with canonical units so the parent-side
    normalization is a no-op; the per-lead value offset makes committed data
    verifiable across waves. No ``tp``/``tcc`` so the 6-hour-reset predecessor
    normalization is out of scope for these tests. ``cycle_time`` mirrors the
    parser's store-identity attribute.
    """
    value = 280.0 + lead
    coords: dict[str, object] = {
        "lead_time_hours": lead,
        "latitude": np.array([38.0, 38.25]),
        "longitude": np.array([-107.0, -106.75]),
    }
    if member is not None:
        coords["member"] = member
    return xr.Dataset(
        data_vars={
            "temperature_2m": (
                ("latitude", "longitude"),
                np.full((2, 2), value, dtype=np.float32),
                {"units": "°C"},
            ),
            "wind_u_10m": (
                ("latitude", "longitude"),
                np.full((2, 2), value - 5.0, dtype=np.float32),
                {"units": "m/s"},
            ),
            "wind_v_10m": (
                ("latitude", "longitude"),
                np.full((2, 2), value - 6.0, dtype=np.float32),
                {"units": "m/s"},
            ),
        },
        coords=coords,  # type: ignore[arg-type]
        attrs={"cycle_time": "2026-07-21T00:00:00", "model_id": "gfs"},
    )


@pytest.fixture()
def stubbed_upstream(monkeypatch: pytest.MonkeyPatch):
    """Stub the download + decode boundaries for offline wave execution."""
    from ingestion.core.decode_worker import DecodePool

    async def _fake_download(
        self, model, cycle_date, cycle_hour, lead_time_hours, destination, member=None, variables=None, **kwargs
    ):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"stub-grib2")
        return destination

    def _fake_submit(self, path):
        name = Path(path).name
        lead = int(name.rsplit(".f", 1)[1].removesuffix(".grib2"))
        member = int(name[3:5]) if name.startswith("gep") else None
        fut: Future = Future()
        fut.set_result(_synthetic_dataset(lead, member))
        return fut

    monkeypatch.setattr(
        "ingestion.providers.noaa.connector.NOAAConnector.download", _fake_download
    )
    monkeypatch.setattr(DecodePool, "submit", _fake_submit)


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        download_dir=str(tmp_path / "dl"),
        keep_downloads=True,
        no_progress=True,
        lock_timeout=5.0,
        center_id="noaa",
        version_string="v1.0",
        grid_id="global_025deg",
        variable=None,
    )


def _run_wave_sync(
    monkeypatch: pytest.MonkeyPatch,
    catalog_db,
    spec: RunSpec,
    args: SimpleNamespace,
    store_path: str,
) -> str:
    """Run one wave with the catalog routed to the test engine; return status."""
    import ingestion.core.wave_runner as wave_runner

    monkeypatch.setattr(wave_runner, "_catalog_session_factory", lambda: catalog_db)
    failures: list[str] = []
    status = asyncio.run(
        _run_wave(
            spec=spec,
            args=args,
            catalog_spec=_build_spec(spec, args, store_path),
            store_path=store_path,
            concurrency=4,
            failures=failures,
        )
    )
    assert failures == [], failures
    return status


def _committed_temperature_regions(store_path: str) -> set[int] | set[tuple[int, int]]:
    """Committed leads (deterministic) / (member, lead) pairs (ensemble).

    Derived from ``temperature_2m`` non-NaN regions — the honest committed-data
    check for these tests (synthetic flag variables are zero-filled and the
    synthetic lead-0 ``tp`` is all-NaN by contract).
    """
    from ingestion.core.zarr_writer import read_dataset

    ds = read_dataset(store_path)
    field = ds["temperature_2m"]
    has = field.notnull().any(dim=("latitude", "longitude"))
    leads = [int(v) for v in ds.coords["lead_time_hours"].values]
    if "member" in field.dims:
        members = [int(v) for v in ds.coords["member"].values]
        return {
            (member, lead)
            for mi, member in enumerate(members)
            for li, lead in enumerate(leads)
            if bool(has.values[mi, li])
        }
    return {lead for li, lead in enumerate(leads) if bool(has.values[li])}


def _run_id(engine, store_path: str) -> str:
    with Session(bind=engine) as db:
        run = db.query(ModelRunRecord).filter_by(zarr_store_path=store_path).one()
        return str(run.id)


def _catalog_pairs(engine, run_id: str) -> set[tuple[int, int]]:
    with Session(bind=engine) as db:
        rows = db.query(EnsembleMemberProductRecord).filter_by(run_id=run_id).all()
        return {(int(r.member_index), int(r.lead_time_hours)) for r in rows}


def _catalog_product_leads(engine, run_id: str) -> set[int]:
    with Session(bind=engine) as db:
        rows = db.query(ProductRecord).filter_by(run_id=run_id).all()
        return {int(r.lead_time_hours) for r in rows}


def _run_status(engine, store_path: str) -> str:
    with Session(bind=engine) as db:
        run = db.query(ModelRunRecord).filter_by(zarr_store_path=store_path).one()
        return str(run.status)


# ---------------------------------------------------------------------------
# _build_spec: the canonical horizon (production default) is wired, not targets
# ---------------------------------------------------------------------------


def test_build_spec_wires_canonical_horizon_not_targets() -> None:
    """_build_spec derives the horizon from the domain contract, never targets."""
    from domain.horizon import canonical_lead_time_hours

    args = _args(Path("."))
    gfs_spec = RunSpec(
        model="gfs",
        cycle_date=CYCLE,
        cycle_hour=0,
        target_lead_time_hours=(0, 3, 6),
    )
    built = _build_spec(
        gfs_spec, args, "s3://weather-data/gfs/2026-07-21/00/cycle.zarr"
    )
    assert built.expected_lead_time_hours == canonical_lead_time_hours("gfs")
    assert len(built.expected_lead_time_hours) == 81  # production default: 0..240 @ 3h
    assert built.expected_members == ()

    gefs_spec = RunSpec(
        model="gefs",
        cycle_date=CYCLE,
        cycle_hour=0,
        target_lead_time_hours=(0,),
        members=(1, 2, 3),  # a member-subset invocation
    )
    built_gefs = _build_spec(
        gefs_spec, args, "s3://weather-data/gefs/2026-07-21/00/cycle.zarr"
    )
    assert built_gefs.expected_lead_time_hours == canonical_lead_time_hours("gefs")
    assert built_gefs.expected_members == tuple(range(1, 31))  # full contract set


def test_run_wave_rejects_targets_outside_canonical_horizon(
    tmp_path: Path, reduced_horizon
) -> None:
    """A wave target outside the canonical horizon is refused before any work."""
    args = _args(tmp_path)
    spec = RunSpec(
        model="gfs",
        cycle_date=CYCLE,
        cycle_hour=0,
        target_lead_time_hours=(HORIZON[0], 300),  # 300 not in the horizon
    )
    with pytest.raises(ValueError, match="outside the canonical cycle horizon"):
        asyncio.run(
            _run_wave(
                spec=spec,
                args=args,
                catalog_spec=_build_spec(spec, args, str(tmp_path / "gfs.zarr")),
                store_path=str(tmp_path / "gfs.zarr"),
                concurrency=1,
                failures=[],
            )
        )


# ---------------------------------------------------------------------------
# Test A — disjoint subset accumulation (GFS + GEFS)
# ---------------------------------------------------------------------------


def test_gfs_disjoint_waves_accumulate_and_stay_partial(
    tmp_path: Path,
    catalog_db,
    reduced_horizon,
    stubbed_upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run1 targets [0,3,6], run2 targets [9,12] -> union committed, run partial."""
    store = str(tmp_path / "gfs.zarr")
    args = _args(tmp_path)

    spec1 = RunSpec(
        model="gfs",
        cycle_date=CYCLE,
        cycle_hour=0,
        target_lead_time_hours=(0, 3, 6),
    )
    status1 = _run_wave_sync(monkeypatch, catalog_db, spec1, args, store)
    assert status1 == "partial"

    # run 2: disjoint targets on the SAME cycle. Before the 5B split this
    # raised StoreSchemaMismatchError (lead axis pre-allocated [0,3,6] only).
    spec2 = RunSpec(
        model="gfs",
        cycle_date=CYCLE,
        cycle_hour=0,
        target_lead_time_hours=(9, 12),
    )
    status2 = _run_wave_sync(monkeypatch, catalog_db, spec2, args, store)
    assert status2 == "partial"

    # Union committed in the store; horizon-incomplete leads still NaN.
    committed = _committed_temperature_regions(store)
    assert committed == {0, 3, 6, 9, 12}

    # Previous leads remain intact (per-lead value offset proves data identity).
    from ingestion.core.zarr_writer import read_dataset

    ds = read_dataset(store)
    assert float(ds["temperature_2m"].sel(lead_time_hours=0).values[0, 0]) == 280.0
    assert float(ds["temperature_2m"].sel(lead_time_hours=9).values[0, 0]) == 289.0

    # The store's lead axis is the full canonical horizon.
    assert sorted(int(v) for v in ds.coords["lead_time_hours"].values) == list(HORIZON)

    # Catalog availability contains the union; the run stays partial.
    run_id = _run_id(catalog_db, store)
    assert _catalog_product_leads(catalog_db, run_id) == {0, 3, 6, 9, 12}
    assert _run_status(catalog_db, store) == "partial"


def test_gefs_disjoint_waves_accumulate_and_stay_partial(
    tmp_path: Path,
    catalog_db,
    reduced_horizon,
    stubbed_upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GEFS waves with disjoint lead targets accumulate (member, lead) pairs."""
    store = str(tmp_path / "gefs.zarr")
    args = _args(tmp_path)
    members = tuple(range(1, 31))

    spec1 = RunSpec(
        model="gefs",
        cycle_date=CYCLE,
        cycle_hour=0,
        target_lead_time_hours=(0, 3),
        members=members,
    )
    status1 = _run_wave_sync(monkeypatch, catalog_db, spec1, args, store)
    assert status1 == "partial"

    spec2 = RunSpec(
        model="gefs",
        cycle_date=CYCLE,
        cycle_hour=0,
        target_lead_time_hours=(6, 9),
        members=members,
    )
    status2 = _run_wave_sync(monkeypatch, catalog_db, spec2, args, store)
    assert status2 == "partial"

    # Union of (member, lead) pairs committed across both waves.
    pairs = _committed_temperature_regions(store)
    assert pairs == {(m, lead) for m in members for lead in (0, 3, 6, 9)}
    assert _catalog_pairs(catalog_db, _run_id(catalog_db, store)) == pairs
    assert _run_status(catalog_db, store) == "partial"

    # Member identity preserved: a wave-2 lead carries real member values.
    from ingestion.core.zarr_writer import read_dataset

    ds = read_dataset(store)
    assert (
        float(ds["temperature_2m"].sel(member=17, lead_time_hours=6).values[0, 0])
        == 286.0
    )


# ---------------------------------------------------------------------------
# Test B — eventual full-horizon convergence (GFS + GEFS)
# ---------------------------------------------------------------------------


def test_gfs_full_horizon_converges_to_ready(
    tmp_path: Path,
    catalog_db,
    reduced_horizon,
    stubbed_upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disjoint waves whose union covers the horizon converge to ready."""
    store = str(tmp_path / "gfs.zarr")
    args = _args(tmp_path)
    waves = [(0, 3), (6, 9), (12, 15), (18, 21)]
    for targets in waves:
        spec = RunSpec(
            model="gfs",
            cycle_date=CYCLE,
            cycle_hour=0,
            target_lead_time_hours=targets,
        )
        status = _run_wave_sync(monkeypatch, catalog_db, spec, args, store)
        assert status == ("ready" if targets == waves[-1] else "partial")

    assert _committed_temperature_regions(store) == set(HORIZON)
    assert _run_status(catalog_db, store) == "ready"
    assert _catalog_product_leads(catalog_db, _run_id(catalog_db, store)) == set(HORIZON)


def test_gefs_full_horizon_converges_to_ready_exact_pairs(
    tmp_path: Path,
    catalog_db,
    reduced_horizon,
    stubbed_upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GEFS: every required (member, lead) pair committed -> run ready.

    Before the 5B split this convergence was impossible: the finalizer
    compared committed pairs against the invoking wave's subset expectation.
    Uses a further-reduced 3-lead GEFS horizon for runtime; the
    ``reduced_horizon`` fixture restores the registry afterwards.
    """
    from domain.horizon import register_canonical_lead_horizon

    horizon3 = (0, 3, 6)
    register_canonical_lead_horizon("gefs", horizon3)

    store = str(tmp_path / "gefs.zarr")
    args = _args(tmp_path)
    members = tuple(range(1, 31))
    for lead in horizon3:
        spec = RunSpec(
            model="gefs",
            cycle_date=CYCLE,
            cycle_hour=0,
            target_lead_time_hours=(lead,),
            members=members,
        )
        status = _run_wave_sync(monkeypatch, catalog_db, spec, args, store)
        assert status == ("ready" if lead == horizon3[-1] else "partial")

    pairs = _committed_temperature_regions(store)
    assert pairs == {(m, lead) for m in members for lead in horizon3}
    assert _catalog_pairs(catalog_db, _run_id(catalog_db, store)) == pairs
    assert _run_status(catalog_db, store) == "ready"
