"""End-to-end shadow validation tests for the sharded_v2 migration tooling.

Builds a real canonical v1 store through the production writer, re-encodes it
into a shadow-v2 namespace via :func:`ingestion.core.shadow.write_shadow_store`,
and asserts the storage/numerical/serving comparison contract:

* storage: shadow payload strictly smaller than the v1 payload;
* numerical: f32 exceptions byte-identical (zero diff, zero threshold flips);
  f16 variables within the measured tolerance;
* serving: point values from both reader classes agree within tolerance and the
  precipitation/ceiling predicates show zero flips;
* cleanup: namespace guard refuses non-shadow prefixes; dry-run default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ingestion.core.config import settings
from ingestion.core.shadow import (
    cleanup_shadow_stores,
    compare_stores_numerical,
    default_shadow_store_path,
    is_shadow_store_path,
    run_shadow_validation,
    write_shadow_store,
)
from ingestion.core.zarr_writer import commit_region, prepare_run_store, read_slice


@pytest.fixture(autouse=True)
def _v1_writer(monkeypatch):
    """The canonical source store is written with the frozen v1 format."""
    monkeypatch.setattr(settings, "STORAGE_FORMAT_VERSION", "sharded_v1")
    yield


def _build_canonical_v1_store(tmp_path: Path) -> tuple[str, dict[str, float]]:
    """Build a realistic multi-variable v1 cycle store (incl. threshold-coupled vars)."""
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.linspace(0.0, 359.75, 1440, dtype=np.float32)
    dims = ("latitude", "longitude")
    shape = (721, 1440)
    rng = np.random.default_rng(11)

    temperature = 15.0 + rng.normal(0, 8, shape).astype(np.float32)
    cloud = np.clip(50.0 + rng.normal(0, 25, shape), 0.0, 100.0).astype(np.float32)
    # Precipitation with real boundary mass: a band of exactly-0.1 mm cells
    # (the GEFS decimal-packing pattern that makes f16 flip the 0.10 threshold)
    # plus a moderate-rain tail.
    precip = np.where(rng.random(shape) < 0.3, np.float32(0.1), np.float32(0.0)).astype(np.float32)
    moderate = rng.random(shape) < 0.05
    precip[moderate] = rng.uniform(0.5, 20.0, int(moderate.sum())).astype(np.float32)
    # Ceiling with sentinel-adjacent structure: unlimited band + near-sentinel band.
    ceiling = np.where(rng.random(shape) < 0.5, np.float32(20.0), np.float32(2.0)).astype(np.float32)
    near_band = rng.random(shape) < 0.002
    ceiling[near_band] = rng.uniform(19.9, 20.0, int(near_band.sum())).astype(np.float32)

    flags = (rng.random(shape) < 0.5).astype(np.float32)

    def ds(lead: int) -> xr.Dataset:
        return xr.Dataset(
            data_vars={
                "temperature_2m": (dims, temperature + lead),
                "cloud_cover_3h": (dims, cloud),
                "precipitation_amount_3h": (dims, precip),
                "cloud_ceiling": (dims, ceiling),
                "crain": (dims, flags),
            },
            coords={
                "lead_time_hours": [lead],
                "latitude": lat,
                "longitude": lon,
                "member": [1],
            },
        )

    store = str(tmp_path / "cycle.zarr")
    prepare_run_store(ds(0), store, expected_lead_time_hours=(0, 6), expected_members=(1,))
    for lead in (0, 6):
        commit_region(ds(lead), store, lead_time_hours=lead, member=1)
    return store, {"temperature_2m": 15.0}


def test_is_shadow_store_path_guard() -> None:
    assert is_shadow_store_path("/x/cycle.shadow-v2.zarr")
    assert is_shadow_store_path("s3://bucket/model/d/18/shadow-v2")
    assert not is_shadow_store_path("s3://bucket/model/d/18/cycle.zarr")
    assert not is_shadow_store_path("/x/cycle.zarr")


def test_default_shadow_store_path() -> None:
    path = default_shadow_store_path("/a/cycle.zarr")
    assert path.replace("\\", "/").endswith("cycle.zarr.shadow-v2")
    assert default_shadow_store_path("s3://b/m/d/18/cycle.zarr").endswith("/shadow-v2")


def test_cleanup_guard_refuses_production_prefix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shadow namespace"):
        cleanup_shadow_stores(str(tmp_path / "cycle.zarr"))
    with pytest.raises(ValueError, match="shadow namespace"):
        cleanup_shadow_stores("s3://weather-data/gfs/2026-09-23/18/cycle.zarr")


def test_cleanup_dry_run_and_age_guard(tmp_path: Path) -> None:
    shadow = tmp_path / "cycle.shadow-v2.zarr"
    shadow.mkdir()
    result = cleanup_shadow_stores(str(shadow), dry_run=True)
    assert result["exists"] is True and result["deleted"] is False and result["dry_run"] is True
    assert shadow.is_dir()  # dry-run leaves it alone

    result = cleanup_shadow_stores(str(shadow), older_than_days=30, dry_run=False)
    assert result.get("skipped") is not None  # age guard: just created
    assert shadow.is_dir()

    result = cleanup_shadow_stores(str(shadow), dry_run=False)
    assert result["deleted"] is True
    assert not shadow.exists()


def test_full_shadow_validation_end_to_end(tmp_path: Path) -> None:
    """The executable GO/NO-GO 2 harness on a local cycle (storage+numerical+serving)."""
    store, _ = _build_canonical_v1_store(tmp_path)
    shadow = default_shadow_store_path(store)
    report = run_shadow_validation(store, shadow, sample_points=25)

    # Storage: v2 payload strictly smaller than v1 for this continuous-heavy mix.
    storage = report["storage"]
    assert storage["shards_reencoded"] > 0
    assert storage["shadow_payload_bytes"] < storage["source_payload_bytes"]
    assert 0.4 < storage["storage_ratio"] < 0.95

    # Numerical: f32 exceptions byte-identical; f16 within tolerance.
    numerical = {r["variable"]: r for r in report["numerical"]}
    precip = numerical["precipitation_amount_3h"]
    assert precip["max_abs_diff"] == 0.0
    assert precip["precip_threshold_flips"] == 0
    ceiling = numerical["cloud_ceiling"]
    assert ceiling["max_abs_diff"] == 0.0
    assert ceiling["ceiling_threshold_flips"] == 0
    temperature = numerical["temperature_2m"]
    assert temperature["max_abs_diff"] <= 0.5
    crain = numerical["crain"]
    assert crain["max_abs_diff"] == 0.0

    # Serving: zero threshold flips through the reader classes.
    assert report["serving"]["per_variable"]["precipitation_amount_3h"]["threshold_flips"] == 0
    assert report["serving"]["per_variable"]["cloud_ceiling"]["threshold_flips"] == 0
    assert report["semantic_regression_free"] is True

    # The shadow store is a complete readable store with a v2 manifest, and the
    # canonical store is untouched (v1 read path unchanged).
    shadow_slice = read_slice(shadow, "temperature_2m", lead_time_hours=6, member=1)
    assert shadow_slice is not None and shadow_slice.dtype == np.dtype("<f2")
    source_slice = read_slice(store, "temperature_2m", lead_time_hours=6, member=1)
    assert source_slice is not None and source_slice.dtype == np.dtype("<f4")

    # Cleanup the shadow namespace (tool-managed lifetime).
    result = cleanup_shadow_stores(shadow, dry_run=False)
    assert result["deleted"] is True


def test_write_shadow_store_refuses_same_path(tmp_path: Path) -> None:
    store, _ = _build_canonical_v1_store(tmp_path)
    with pytest.raises(ValueError, match="must differ"):
        write_shadow_store(store, store)


def test_compare_stores_numerical_tolerance_contract(tmp_path: Path) -> None:
    store, _ = _build_canonical_v1_store(tmp_path)
    shadow = default_shadow_store_path(store)
    write_shadow_store(store, shadow)
    reports = {r.variable: r for r in compare_stores_numerical(store, shadow)}
    # f32 exception: zero tolerance.
    assert reports["precipitation_amount_3h"].tolerance == 0.0
    assert reports["cloud_ceiling"].tolerance == 0.0
    # f16 continuous: generous regression bound.
    assert reports["temperature_2m"].tolerance == 0.5
    assert reports["cloud_cover_3h"].n_slices_compared == 2
