"""Cross-process region-write concurrency E2E tests (real PostgreSQL + MinIO).

Two independent OS processes write the same run store through the approved
coordinator protocol. The PostgreSQL advisory locks provide cross-process
coordination. Processes are spawned via ``subprocess.Popen`` (true process
isolation; each child creates its own SQLAlchemy Engine/Connection).
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../packages/domain/src")))

from ingestion.core.config import settings

_WORKER = r"""
import sys, os, threading, time
sys.path.insert(0, %r)
sys.path.insert(0, %r)
from datetime import datetime, timezone
from sqlalchemy import create_engine
from ingestion.core.config import settings
from ingestion.core.catalog import RunCatalogSpec, VariableSpec
from ingestion.core.coordinator import RunCoordinator, WaveRegion
import numpy as np, xarray as xr

lead = int(sys.argv[1])
member_raw = sys.argv[2]
s3_store = sys.argv[3]
barrier_file = sys.argv[4]
barrier_count = int(sys.argv[5])
member = int(member_raw) if member_raw != "None" else None

spec = RunCatalogSpec(
    center_id="noaa", center_name="NOAA", center_country="USA",
    model_id="gfs", model_name="GFS",
    is_ensemble=member is not None, resolution_km=25.0,
    version_string="v1.0",
    cycle_time=datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc),
    grid_id="global_025deg", grid_name="g", grid_resolution_km=25.0,
    zarr_store_path=s3_store,
    variables=(VariableSpec("temperature_2m", "T", "degC", "t2m"),),
    expected_lead_time_hours=(6, 12),
    # Ensemble members are 1..n_writers (derived from the barrier count) so a
    # 4-way test initializes the store for members 1..4.
    expected_members=tuple(range(1, barrier_count + 1)) if member is not None else (),
)
lat = np.array([38.0, 38.25, 38.5, 38.75])
lon = np.array([-107.0, -106.75, -106.5, -106.25])
dims = ("lead_time_hours", "latitude", "longitude")
shape = (1, 4, 4)
coords = {"lead_time_hours": [lead], "latitude": lat, "longitude": lon,
          "time": np.datetime64("2026-07-22T00:00:00")}
if member is not None:
    dims = ("member", "lead_time_hours", "latitude", "longitude")
    shape = (1, 1, 4, 4)
    coords["member"] = [member]
# Each member writes a distinct sentinel (member identity) so lost updates are
# detectable; deterministic writes use the lead value.
_write_value = float(member) if member is not None else float(lead)
ds = xr.Dataset(
    data_vars={"temperature_2m": (dims, np.full(shape, _write_value, dtype=np.float32))},
    coords=coords, attrs={"cycle_time": "2026-07-22T00:00:00", "model_id": "gfs"},
)
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
coordinator = RunCoordinator(spec, s3_store, timeout_seconds=10.0)
cancel_event = threading.Event()

# Seed region initializes the store: the deterministic lead-6 worker, OR the
# ensemble member-1 worker (so a member-only ensemble test still initializes).
if (lead == 6 and member is None) or member == 1:
    conn = engine.connect()
    try:
        coordinator.initialize_run_store(
            conn, seed_dataset=ds, expected_leads=(6, 12),
            expected_members=tuple(range(1, barrier_count + 1)) if member is not None else (),
            run_id=None, is_same_cycle=False,
        )
    finally:
        conn.close()

# Barrier: append this process's pid, wait for both.
with open(barrier_file, "a", encoding="utf-8") as fh:
    fh.write(str(os.getpid()) + "\n")
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    try:
        with open(barrier_file, "r", encoding="utf-8") as fh:
            n = len(fh.read().strip().splitlines())
        if n >= barrier_count:
            break
    except OSError:
        pass
    time.sleep(0.05)

conn = engine.connect()
try:
    from concurrent.futures import ThreadPoolExecutor
    gen = __import__("uuid").uuid4().hex
    regions = [WaveRegion(lead_time_hours=lead, member=member, generation=gen)]
    coordinator.pre_update_wave(
        conn, regions=regions, run_id=None, is_same_cycle=False,
        executor=ThreadPoolExecutor(1), cancel_event=cancel_event,
    )
    coordinator.write_region_worker(
        conn, dataset=ds, member=member, generation=gen,
        expected_leads=(6, 12), expected_members=tuple(range(1, barrier_count + 1)) if member is not None else (),
    )
finally:
    conn.close()
    engine.dispose()
"""


def _ensure_catalog_schema() -> None:
    """Create the ingestion catalog schema on the real PostgreSQL engine."""
    from sqlalchemy import create_engine

    from ingestion.core.catalog import CatalogBase

    engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
    try:
        CatalogBase.metadata.create_all(engine)
    finally:
        engine.dispose()


def _minio_reachable() -> bool:
    from minio import Minio  # type: ignore[import-untyped]

    try:
        client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        client.bucket_exists(settings.MINIO_BUCKET_NAME)
        return True
    except Exception:
        return False


def _spawn_worker(lead: int, member: int | None, store: str, barrier: str, n: int):
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
    domain_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../packages/domain/src"))
    script = _WORKER % (src_dir, domain_src)
    cmd = [
        sys.executable, "-c", script,
        str(lead), "None" if member is None else str(member), store, barrier, str(n),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.mark.skipif(
    os.environ.get("WEATHER_TEST_MINIO") != "1" or not _minio_reachable(),
    reason="MinIO integration test",
)
def test_cross_process_disjoint_deterministic_writes() -> None:
    """Two processes write disjoint leads 6 and 12; both survive; union in store."""
    _ensure_catalog_schema()
    store = f"s3://{settings.MINIO_BUCKET_NAME}/m5-cross/{uuid.uuid4().hex}"
    barrier = os.path.join(os.path.dirname(__file__), "_m5-barrier")
    if os.path.exists(barrier):
        os.unlink(barrier)

    p_a = _spawn_worker(6, None, store, barrier, 2)
    p_b = _spawn_worker(12, None, store, barrier, 2)
    _, err_a = p_a.communicate(timeout=90)
    _, err_b = p_b.communicate(timeout=90)
    assert p_a.returncode == 0, f"process A (lead 6) failed: {err_a.decode()[:500]}"
    assert p_b.returncode == 0, f"process B (lead 12) failed: {err_b.decode()[:500]}"

    from ingestion.core.zarr_writer import read_dataset

    ds = read_dataset(store)
    assert sorted(int(v) for v in ds.coords["lead_time_hours"].values) == [6, 12]
    v6 = float(ds["temperature_2m"].sel(lead_time_hours=6).values[0, 0])
    v12 = float(ds["temperature_2m"].sel(lead_time_hours=12).values[0, 0])
    assert v6 == pytest.approx(6.0)
    assert v12 == pytest.approx(12.0)
    # Both COMPLETE markers survive (the finalizer can catalog the union).
    from ingestion.core.markers import list_region_marker_keys

    keys = list_region_marker_keys(store)
    assert len(keys) == 2, f"expected 2 COMPLETE markers, got {keys}"
    if os.path.exists(barrier):
        os.unlink(barrier)


@pytest.mark.skipif(
    os.environ.get("WEATHER_TEST_MINIO") != "1" or not _minio_reachable(),
    reason="MinIO integration test",
)
def test_cross_process_conflicting_ensemble_serializes() -> None:
    """Two processes target different members of the SAME lead -> serialize."""
    _ensure_catalog_schema()
    store = f"s3://{settings.MINIO_BUCKET_NAME}/m5-cross/{uuid.uuid4().hex}"
    barrier = os.path.join(os.path.dirname(__file__), "_m5-barrier-ens")
    if os.path.exists(barrier):
        os.unlink(barrier)

    p_a = _spawn_worker(6, 1, store, barrier, 2)
    p_b = _spawn_worker(6, 2, store, barrier, 2)
    _, err_a = p_a.communicate(timeout=90)
    _, err_b = p_b.communicate(timeout=90)
    assert p_a.returncode == 0, f"process A (member 1) failed: {err_a.decode()[:500]}"
    assert p_b.returncode == 0, f"process B (member 2) failed: {err_b.decode()[:500]}"

    from ingestion.core.zarr_writer import read_dataset

    ds = read_dataset(store)
    assert "member" in ds.coords
    m1 = float(ds["temperature_2m"].sel(member=1, lead_time_hours=6).values[0, 0])
    m2 = float(ds["temperature_2m"].sel(member=2, lead_time_hours=6).values[0, 0])
    # Members write distinct sentinels (member identity) to detect lost updates.
    assert m1 == pytest.approx(1.0)
    assert m2 == pytest.approx(2.0)
    if os.path.exists(barrier):
        os.unlink(barrier)


@pytest.mark.skipif(
    os.environ.get("WEATHER_TEST_MINIO") != "1" or not _minio_reachable(),
    reason="MinIO integration test",
)
def test_cross_process_four_way_conflict_no_lost_members() -> None:
    """Four independent writers target members 1..4 of the SAME lead. Each writes
    a distinct sentinel. Under the full-member chunk layout all four share the
    physical chunk, so max_active == 1 and no member value is lost."""
    _ensure_catalog_schema()
    store = f"s3://{settings.MINIO_BUCKET_NAME}/m5-cross/{uuid.uuid4().hex}"
    barrier = os.path.join(os.path.dirname(__file__), "_m5-barrier-4way")
    if os.path.exists(barrier):
        os.unlink(barrier)

    procs = [_spawn_worker(6, m, store, barrier, 4) for m in (1, 2, 3, 4)]
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, f"process failed: {err.decode()[:500]}"

    from ingestion.core.zarr_writer import read_dataset

    ds = read_dataset(store)
    lost = []
    for m in (1, 2, 3, 4):
        v = float(ds["temperature_2m"].sel(member=m, lead_time_hours=6).values[0, 0])
        if v != float(m):
            lost.append((m, v))
    assert not lost, f"lost member values: {lost}"
    if os.path.exists(barrier):
        os.unlink(barrier)
