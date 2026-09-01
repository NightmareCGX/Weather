"""Unit tests for intermediate settled-lead publication in ingestion coordinator."""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytest
import xarray as xr
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ingestion.core.inventory import expected_write_set_fingerprint
from ingestion.core.catalog import (
    CatalogBase,
    EnsembleMemberProductRecord,
    ModelRunRecord,
    ProductRecord,
    RunCatalogSpec,
    VariableSpec,
    record_run,
)
from ingestion.core.coordinator import RunCoordinator
from ingestion.core.markers import MARKER_V1, write_protocol_version, write_region_marker
from ingestion.core.zarr_writer import prepare_run_store


class _NoopStoreLockCoordinator:
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


@pytest.fixture(autouse=True)
def stub_advisory_locks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub PostgreSQL advisory locks so coordinator tests run against SQLite."""
    monkeypatch.setattr(
        "ingestion.core.coordinator.StoreLockCoordinator",
        _NoopStoreLockCoordinator,
    )


@pytest.fixture
def temp_ingest_db(tmp_path: Path):
    """Temporary SQLite database session for ingestion coordinator tests."""
    db_path = tmp_path / "ingest_test.db"
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)
    CatalogBase.metadata.create_all(engine)
    session = Session(engine)
    yield session, engine, db_url
    session.close()
    engine.dispose()


def test_publish_settled_lead_creates_catalog_and_advances_generation(temp_ingest_db, tmp_path: Path) -> None:
    session, engine, db_url = temp_ingest_db
    store_dir = tmp_path / "gefs_cycle.zarr"
    store_path = str(store_dir)

    cycle_time = datetime(2026, 8, 14, 6, 0, tzinfo=timezone.utc)
    expected_leads = (0, 3, 6)
    expected_members = tuple(range(1, 31))

    # Initialize store
    seed_ds = xr.Dataset(
        data_vars={
            "temperature_2m": (("latitude", "longitude"), np.ones((2, 2))),
        },
        coords={
            "latitude": [38.0, 38.25],
            "longitude": [-107.0, -106.75],
            "lead_time_hours": [0],
            "member": [1],
        },
    )
    prepare_run_store(seed_ds, store_path, expected_lead_time_hours=expected_leads, expected_members=expected_members)
    write_protocol_version(store_path, MARKER_V1)

    spec = RunCatalogSpec(
        center_id="noaa",
        center_name="NOAA",
        center_country="USA",
        model_id="gefs",
        model_name="GEFS",
        is_ensemble=True,
        resolution_km=25.0,
        version_string="v1.0",
        cycle_time=cycle_time,
        grid_id="global_025deg",
        grid_name="0.25 deg",
        grid_resolution_km=25.0,
        zarr_store_path=store_path,
        variables=(
            VariableSpec(code="temperature_2m", name="2m Temp", unit="°C"),
        ),
        expected_lead_time_hours=expected_leads,
        expected_members=expected_members,
    )

    # Initial record_run
    run_record = record_run(session, spec, seed_ds, member=1)
    run_id = run_record.id
    assert run_record.status in ("processing", "partial")

    coordinator = RunCoordinator(spec=spec, store_path=store_path, endpoint="", secure=False)
    lead_index = coordinator._lead_index_for(0)
    data_var_paths = ["temperature_2m"]

    # Write COMPLETE markers for lead 0 (28 members out of 30)
    from ingestion.core.inventory import region_expected_object_keys
    for m in range(1, 29):
        expected_keys = region_expected_object_keys(
            store_path,
            member=m,
            lead_index=lead_index,
            data_var_paths=data_var_paths,
            zarray_cache=coordinator._zarray_cache,
            zattrs_cache=coordinator._zattrs_cache,
            member_index_cache=coordinator._member_index_cache,
        )
        fp = expected_write_set_fingerprint(expected_keys, [])
        write_region_marker(
            store_path,
            lead_time_hours=0,
            member=m,
            payload={
                "protocol_version": 1,
                "state": "complete",
                "generation": "gen_test",
                "logical_region": {"lead_time_hours": 0, "member": m},
                "expected_write_set_fingerprint": fp,
                "required_materialized_object_keys": expected_keys,
                "intentionally_omitted_fill_chunks": [],
            },
        )

    with engine.connect() as conn:
        coordinator.publish_settled_lead(
            conn,
            run_id=run_id,
            spec=spec,
            lead_time_hours=0,
            expected_members=expected_members,
        )

    # Verify catalog rows were committed for lead 0
    with Session(engine) as db:
        run = db.get(ModelRunRecord, run_id)
        assert run is not None
        # Full run status must NOT be marked ready (remains processing/partial)
        assert run.status in ("processing", "partial")

        # Check ensemble_member_products for lead 0
        emp_rows = db.execute(
            select(EnsembleMemberProductRecord.member_index).where(
                EnsembleMemberProductRecord.run_id == run_id,
                EnsembleMemberProductRecord.lead_time_hours == 0,
            )
        ).scalars().all()
        assert len(emp_rows) == 28
        assert set(emp_rows) == set(range(1, 29))

        # Check forecast_products for lead 0
        fp_rows = db.execute(
            select(ProductRecord.lead_time_hours).where(
                ProductRecord.run_id == run_id,
                ProductRecord.variable_id == "temperature_2m",
                ProductRecord.lead_time_hours == 0,
            )
        ).scalars().all()
        assert len(fp_rows) == 1
