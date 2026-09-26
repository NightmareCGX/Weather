"""Cross-package contract: sharded_v2 shadow validation with serving comparison.

Runs in the ``uv sync --all-packages`` contract job, the only CI environment
that installs both the ingestion and api packages together. Asserts the full
producer/consumer compatibility of the shadow-validation tooling:

* ingestion writer (v1 canonical) -> shadow re-encode (v2) -> numerical
  comparison: f32 semantic exceptions (precipitation exactly-0.10 mm,
  cloud_ceiling 19.99 km sentinel) byte-identical with ZERO predicate flips;
* serving comparison through BOTH production reader classes
  (ShardedV1Reader + ShardedV2Reader, dispatched per store manifest): point
  values agree within the float16 storage tolerance and both threshold
  predicates show zero flips;
* the operator CLI's verdict logic (``scripts/shadow_v2.py``) reports the
  validation as semantic-regression-free.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts" / "shadow_v2.py"

from ingestion.core.config import settings  # noqa: E402
from ingestion.core.shadow import (  # noqa: E402
    cleanup_shadow_stores,
    compare_stores_numerical,
    default_shadow_store_path,
    write_shadow_store,
)
from ingestion.core.zarr_writer import (  # noqa: E402
    commit_region,
    prepare_run_store,
    read_slice,
)


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("shadow_v2_cli", SCRIPTS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("shadow_v2_cli", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _v1_writer(monkeypatch):
    """The canonical source store is written with the frozen v1 format."""
    monkeypatch.setattr(settings, "STORAGE_FORMAT_VERSION", "sharded_v1")
    yield


def _build_canonical_v1_store(tmp_path: Path) -> str:
    """A multi-variable v1 cycle store with real threshold-coupled structure."""
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.linspace(0.0, 359.75, 1440, dtype=np.float32)
    dims = ("latitude", "longitude")
    shape = (721, 1440)
    rng = np.random.default_rng(23)

    temperature = 15.0 + rng.normal(0, 8, shape).astype(np.float32)
    cloud = np.clip(50.0 + rng.normal(0, 25, shape), 0.0, 100.0).astype(np.float32)
    # Exactly-0.1 mm drizzle band: the GEFS decimal-packing pattern that makes
    # float16 storage flip the 0.10 mm threshold (hence the f32 exception).
    precip = np.where(rng.random(shape) < 0.3, np.float32(0.1), np.float32(0.0)).astype(
        np.float32
    )
    moderate = rng.random(shape) < 0.05
    precip[moderate] = rng.uniform(0.5, 20.0, int(moderate.sum())).astype(np.float32)
    # Ceiling sentinel band with near-threshold values (19.99 km boundary).
    ceiling = np.where(rng.random(shape) < 0.5, np.float32(20.0), np.float32(3.0)).astype(
        np.float32
    )
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
    return store


def test_shadow_validation_serving_contract(tmp_path: Path) -> None:
    cli = _load_cli_module()
    store = _build_canonical_v1_store(tmp_path)
    shadow = default_shadow_store_path(store)

    report = cli.run_shadow_validation(store, shadow, sample_points=25)

    # Storage: v2 shadow payload strictly smaller than the v1 canonical payload.
    assert report["storage"]["shards_reencoded"] > 0
    assert report["storage"]["shadow_payload_bytes"] < report["storage"]["source_payload_bytes"]

    # Numerical: f32 exceptions byte-identical, zero predicate flips.
    numerical = {r["variable"]: r for r in report["numerical"]}
    assert numerical["precipitation_amount_3h"]["max_abs_diff"] == 0.0
    assert numerical["precipitation_amount_3h"]["precip_threshold_flips"] == 0
    assert numerical["cloud_ceiling"]["max_abs_diff"] == 0.0
    assert numerical["cloud_ceiling"]["ceiling_threshold_flips"] == 0

    # Serving: both production reader classes agree; zero threshold flips.
    serving = report["serving"]["per_variable"]
    assert serving["precipitation_amount_3h"]["threshold_flips"] == 0
    assert serving["cloud_ceiling"]["threshold_flips"] == 0
    assert serving["temperature_2m"]["threshold_flips"] == 0
    assert serving["temperature_2m"]["storage_dtype"] == "<f2"
    assert serving["temperature_2m"]["max_abs_diff"] <= 0.5

    # Verdict: the CLI reports the migration as semantic-regression-free.
    assert report["semantic_regression_free"] is True

    # The v2 shadow store serves through the native dtype path; the canonical
    # v1 store keeps its frozen float32 readback.
    shadow_slice = read_slice(shadow, "temperature_2m", lead_time_hours=6, member=1)
    assert shadow_slice is not None and shadow_slice.dtype == np.dtype("<f2")
    source_slice = read_slice(store, "temperature_2m", lead_time_hours=6, member=1)
    assert source_slice is not None and source_slice.dtype == np.dtype("<f4")

    result = cleanup_shadow_stores(shadow, dry_run=False)
    assert result["deleted"] is True
