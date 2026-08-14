"""Unit tests for the ingestion PostgreSQL catalog writer.

``record_run`` is tested against an in-memory SQLite database (no live
PostgreSQL required), verifying row creation, idempotent upserts, the
``ready`` run state, forecast-product rows per (variable, lead), and
ensemble-member rows for datasets with a ``member`` dimension.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ingestion.core.catalog import (
    CatalogBase,
    CenterRecord,
    EnsembleMemberRecord,
    GridRecord,
    ModelRecord,
    ModelRunRecord,
    ModelVersionRecord,
    ProductRecord,
    RunCatalogSpec,
    VariableRecord,
    VariableSpec,
    record_ingested_dataset,
    record_run,
)

CYCLE = datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc)

TEMPERATURE = VariableSpec("temperature_2m", "2-Meter Temperature", "°C")
PRECIPITATION = VariableSpec("precipitation_rate", "Precipitation Rate", "mm/h")


def _dataset(*, with_member: bool = False) -> xr.Dataset:
    """A deterministic dataset with a 2-element lead coordinate."""
    dims = ("lead_time_hours", "latitude", "longitude")
    if with_member:
        dims = ("member", "lead_time_hours", "latitude", "longitude")
        shape = (3, 2, 2, 2)
        coords = {
            "member": [0, 1, 2],
            "lead_time_hours": [0, 6],
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
        }
    else:
        shape = (2, 2, 2)
        coords = {
            "lead_time_hours": [0, 6],
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 1.0],
        }
    return xr.Dataset(
        data_vars={
            "temperature_2m": (dims, np.ones(shape, dtype=float)),
            "precipitation_rate": (dims, np.ones(shape, dtype=float)),
        },
        coords=coords,
    )


def _spec(**overrides: object) -> RunCatalogSpec:
    base: dict[str, object] = {
        "center_id": "noaa",
        "center_name": "National Oceanic and Atmospheric Administration",
        "center_country": "USA",
        "model_id": "gfs",
        "model_name": "Global Forecast System",
        "is_ensemble": False,
        "resolution_km": 25.0,
        "version_string": "v1.0",
        "cycle_time": CYCLE,
        "grid_id": "global_025deg",
        "grid_name": "Global 0.25 Degree Grid",
        "grid_resolution_km": 25.0,
        "product_type": "surface",
        "zarr_store_path": "/tmp/gfs.zarr",
        "variables": (TEMPERATURE, PRECIPITATION),
    }
    base.update(overrides)
    return RunCatalogSpec(**base)


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


def test_record_run_creates_all_catalog_rows(session: Session) -> None:
    run = record_run(session, _spec(), _dataset())

    assert run.status == "ready"
    assert run.zarr_store_path == "/tmp/gfs.zarr"
    assert session.query(ModelRunRecord).count() == 1
    assert session.query(ModelVersionRecord).count() == 1
    assert session.query(ModelRecord).count() == 1
    assert session.query(CenterRecord).count() == 1
    assert session.query(GridRecord).count() == 1
    assert session.query(VariableRecord).count() == 2
    # One product per (variable x lead): 2 variables x 2 leads.
    assert session.query(ProductRecord).count() == 4
    assert session.query(EnsembleMemberRecord).count() == 0


def test_record_run_sets_ready_and_discoverable_state(session: Session) -> None:
    record_run(session, _spec(), _dataset())
    run = session.query(ModelRunRecord).one()
    assert run.status == "ready"
    assert run.zarr_store_path == "/tmp/gfs.zarr"
    # The API _resolve_run query shape: ready + non-null store path.
    assert run.status == "ready" and run.zarr_store_path is not None


def test_record_run_idempotent_upsert(session: Session) -> None:
    spec = _spec()
    first = record_run(session, spec, _dataset())
    second = record_run(session, spec, _dataset())

    assert first.id == second.id
    assert session.query(ModelRunRecord).count() == 1
    assert session.query(ModelVersionRecord).count() == 1
    assert session.query(ModelRecord).count() == 1
    assert session.query(ProductRecord).count() == 4
    assert session.query(EnsembleMemberRecord).count() == 0


def test_record_run_ensemble_members(session: Session) -> None:
    run = record_run(
        session,
        _spec(is_ensemble=True, model_id="gefs"),
        _dataset(with_member=True),
    )

    members = session.query(EnsembleMemberRecord).all()
    assert len(members) == 3
    assert {member.member_index for member in members} == {0, 1, 2}
    assert all(member.run_id == run.id for member in members)
    # Products still keyed by (variable x lead), independent of members.
    assert session.query(ProductRecord).count() == 4


def test_record_run_products_cover_each_variable_and_lead(session: Session) -> None:
    record_run(session, _spec(), _dataset())
    rows = session.query(ProductRecord).all()
    pairs = {(row.variable_id, row.lead_time_hours) for row in rows}
    assert pairs == {
        ("temperature_2m", 0),
        ("temperature_2m", 6),
        ("precipitation_rate", 0),
        ("precipitation_rate", 6),
    }
    assert all(row.product_type == "surface" for row in rows)


def test_record_run_naive_cycle_time_normalized_to_utc(session: Session) -> None:
    spec = _spec(cycle_time=datetime(2026, 7, 21, 0, 0))  # naive
    run = record_run(session, spec, _dataset())
    # The run id is derived from the UTC-normalized cycle time.
    assert run.id == "run_version_gfs_v1.0_202607210000_gfs"


def test_record_run_update_existing_run_store_path(session: Session) -> None:
    # Re-record with a different store path must refresh the same run.
    record_run(session, _spec(zarr_store_path="/tmp/gfs_v1.zarr"), _dataset())
    second = record_run(session, _spec(zarr_store_path="/tmp/gfs_v2.zarr"), _dataset())
    assert second.id == session.query(ModelRunRecord).one().id
    assert second.zarr_store_path == "/tmp/gfs_v2.zarr"
    assert session.query(ModelRunRecord).count() == 1


def test_record_run_counts_unique_run_model_version(session: Session) -> None:
    # Two runs with the same model+version but different cycle times.
    run_a = record_run(session, _spec(), _dataset())
    run_b = record_run(
        session, _spec(cycle_time=datetime(2026, 7, 21, 12, tzinfo=timezone.utc)),
        _dataset(),
    )
    assert run_a.id != run_b.id
    assert session.query(ModelRunRecord).count() == 2
    assert session.query(ModelVersionRecord).count() == 1


def test_record_run_version_scoped_ids_no_pk_collision(session: Session) -> None:
    """Two versions of the same model at the same cycle get distinct run ids.

    The run id is version-scoped (GAP-4 fix): the schema's
    ``(model_version_id, cycle_time)`` uniqueness allows both rows, so the id
    must distinguish them or the second insert collides on the primary key.
    """
    run_a = record_run(
        session, _spec(version_string="v1.0"), _dataset()
    )
    run_b = record_run(
        session, _spec(version_string="v2.0"), _dataset()
    )
    assert run_a.id != run_b.id
    assert session.query(ModelRunRecord).count() == 2
    # Both versions share the same model row but are distinct versions.
    assert session.query(ModelVersionRecord).count() == 2
    assert session.query(ModelRecord).count() == 1
    assert "v1.0" in run_a.id
    assert "v2.0" in run_b.id


def test_ensemble_members_unique_across_runs(session: Session) -> None:
    # Two runs of the same ensemble model must not collide on member PK ids.
    first = record_run(
        session,
        _spec(is_ensemble=True, model_id="gefs"),
        _dataset(with_member=True),
    )
    second = record_run(
        session,
        _spec(
            is_ensemble=True,
            model_id="gefs",
            cycle_time=datetime(2026, 7, 21, 12, tzinfo=timezone.utc),
        ),
        _dataset(with_member=True),
    )
    assert first.id != second.id
    rows = session.query(EnsembleMemberRecord).all()
    assert len(rows) == 6
    assert len({row.id for row in rows}) == 6  # no PK collisions


def test_record_ingested_dataset_retries_on_integrity_error(
    monkeypatch, tmp_path
) -> None:
    """A concurrent uniqueness collision on the run row is retried instead of
    surfacing an IntegrityError, and the effective store path is recorded."""
    engine = create_engine("sqlite:///:memory:")
    CatalogBase.metadata.create_all(engine)

    def _session_local():
        return Session(engine)

    monkeypatch.setattr("ingestion.core.catalog.SessionLocal", _session_local)
    monkeypatch.setattr("ingestion.core.db.SessionLocal", _session_local)

    spec = _spec(zarr_store_path="/tmp/gfs.zarr")
    calls = {"n": 0}

    def _run_with_one_collision(db, s, ds):
        calls["n"] += 1
        if calls["n"] == 1:
            # Simulate a concurrent worker committing the same run between
            # our SELECT and INSERT on the first attempt.
            from sqlalchemy.exc import IntegrityError

            record_run(db, _spec(), ds)  # commits the run row
            raise IntegrityError("stmt", {}, Exception("unique violation"))
        return record_run(db, s, ds)

    monkeypatch.setattr(
        "ingestion.core.catalog.record_run", _run_with_one_collision
    )

    run = record_ingested_dataset(spec, _dataset(), effective_store_path="/tmp/gfs.zarr")
    assert run.id == "run_version_gfs_v1.0_202607210000_gfs"
    assert calls["n"] == 2  # first collided, second succeeded
    assert run.zarr_store_path == "/tmp/gfs.zarr"
