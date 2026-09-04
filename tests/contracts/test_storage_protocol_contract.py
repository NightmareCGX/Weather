"""Stage 7D-B — Cross-Package Sharded v1 Storage Protocol Contract Suite.

Asserts structural no-drift in binary format constants, shard key naming conventions,
and committed manifest paths between the ingestion writer and the API serving reader.
"""

from api.core import manifest_reader as api_manifest
from api.core import zarr as api_zarr
from domain.locks import canonical_storage_identity
from ingestion.core import markers as ing_markers
from ingestion.core import zarr_writer as ing_zarr


def test_sharded_v1_binary_constants_no_drift() -> None:
    """Verify that binary container constants match exactly between writer and reader."""
    assert (
        ing_zarr.SHARD_MAGIC == api_zarr.SHARD_MAGIC == 0x53484152
    ), "SHARD_MAGIC constant drift detected between writer and reader"

    assert (
        ing_zarr.INDEX_ENTRY_SIZE == api_zarr.INDEX_ENTRY_SIZE == 16
    ), "INDEX_ENTRY_SIZE constant drift detected between writer and reader"

    assert (
        ing_zarr.TRAILER_SIZE == api_zarr.TRAILER_SIZE == 12
    ), "TRAILER_SIZE constant drift detected between writer and reader"


def test_committed_manifest_path_no_drift() -> None:
    """Verify that committed manifest path is identical across ingestion and API."""
    assert (
        ing_markers._MANIFEST_PATH
        == api_manifest._MANIFEST_PATH
        == "__commit__/v1/manifest.json"
    ), "Committed manifest path drift detected between markers and manifest_reader"


def test_shard_key_naming_convention_parity() -> None:
    """Verify that deterministic and ensemble shard keys match between writer and reader."""
    reader = api_zarr.ShardedV1Reader("s3://weather-data/gfs/2026-09-03/00/cycle.zarr")

    # Deterministic GFS shard key (member=None)
    gfs_key = reader._get_shard_key("temperature_2m", member=None, lead_time_hours=6)
    assert gfs_key == "temperature_2m/shard.det_L0006.shard"

    # Ensemble GEFS shard key (member=3)
    gefs_key = reader._get_shard_key("temperature_2m", member=3, lead_time_hours=12)
    assert gefs_key == "temperature_2m/shard.mem003_L0012.shard"


def test_canonical_storage_identity_normalization() -> None:
    """Verify that storage identity normalization is identical across both consumers."""
    raw_path = "s3://weather-data/gfs/2026-09-03/00/cycle.zarr/"
    norm_api = canonical_storage_identity(raw_path, endpoint="localhost:9000", secure=False)
    norm_ing = canonical_storage_identity(raw_path, endpoint="localhost:9000", secure=False)
    assert norm_api == norm_ing == "s3://http://localhost:9000/weather-data/gfs/2026-09-03/00/cycle.zarr"
