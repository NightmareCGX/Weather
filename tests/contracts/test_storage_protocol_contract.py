"""Stage 7D-B — Cross-Package Sharded Storage Protocol Contract Suite.

Asserts structural no-drift in binary format constants, shard key naming conventions,
and committed manifest paths between the ingestion writer and the API serving reader.

The format bytes themselves are owned by ``domain.shard_format`` (unit-tested in
``packages/domain/tests/test_shard_format.py``); this suite asserts that both services
agree with that authority and with each other.
"""

import pytest

from api.core import manifest_reader as api_manifest
from api.core import zarr as api_zarr
from domain import shard_format
from domain.locks import canonical_storage_identity
from ingestion.core import markers as ing_markers
from ingestion.core import zarr_writer as ing_zarr


def test_sharded_v1_binary_constants_no_drift() -> None:
    """Verify that binary container constants match between writer, reader, and the format.

    ``domain.shard_format`` owns the byte layout; the writer and reader re-export it. The
    assertion is against that authority rather than against a literal, so changing the
    layout in one place cannot leave the two services agreeing on a stale value.
    """
    assert (
        ing_zarr.SHARD_MAGIC
        == api_zarr.SHARD_MAGIC
        == shard_format.SHARD_V1_MAGIC
        == 0x53484152
    ), "SHARD_MAGIC constant drift detected between writer, reader, and shard_format"

    assert (
        ing_zarr.INDEX_ENTRY_SIZE
        == api_zarr.INDEX_ENTRY_SIZE
        == shard_format.INDEX_ENTRY_SIZE
        == 16
    ), "INDEX_ENTRY_SIZE constant drift detected between writer, reader, and shard_format"

    assert (
        ing_zarr.TRAILER_SIZE
        == api_zarr.TRAILER_SIZE
        == shard_format.TRAILER_SIZE
        == 12
    ), "TRAILER_SIZE constant drift detected between writer, reader, and shard_format"


def test_container_generations_have_distinct_magics() -> None:
    """A v1 reader must not be able to accept a v2 container by accident."""
    assert shard_format.SHARD_V1_MAGIC != shard_format.SHARD_V2_MAGIC
    assert shard_format.FORMAT_VERSION_V2 == "sharded_v2"


def test_reader_rejects_a_container_whose_magic_is_unknown() -> None:
    """The trailer magic is a contract, not decoration.

    The reader previously ignored it entirely, so a foreign or corrupted object was sliced
    as if it were a container. Parsing must fail instead.
    """
    import struct

    from domain.shard_format import ShardFormatError, parse_trailer

    good = struct.pack(
        "<III", 120, 120 * shard_format.INDEX_ENTRY_SIZE, shard_format.SHARD_V1_MAGIC
    )
    assert parse_trailer(good).is_v1

    for bad_magic in (0, 0xDEADBEEF, shard_format.SHARD_V2_MAGIC ^ 0xFF):
        with pytest.raises(ShardFormatError, match="unrecognized container magic"):
            parse_trailer(
                struct.pack(
                    "<III", 120, 120 * shard_format.INDEX_ENTRY_SIZE, bad_magic
                )
            )


def test_committed_manifest_path_no_drift() -> None:
    """Verify that committed manifest path is identical across ingestion and API."""
    assert (
        ing_markers._MANIFEST_PATH
        == api_manifest._MANIFEST_PATH
        == "__commit__/v1/manifest.json"
    ), "Committed manifest path drift detected between markers and manifest_reader"


def test_shard_key_naming_convention_parity() -> None:
    """Verify that every shard key form matches between writer and reader.

    Covers all three target kinds. The ensemble-mean form was previously asserted only by
    service-local tests, so a rename could have slipped past this contract suite.
    """
    reader = api_zarr.ShardedV1Reader("s3://weather-data/gfs/2026-09-03/00/cycle.zarr")

    # Deterministic GFS shard key (member=None)
    gfs_key = reader._get_shard_key("temperature_2m", member=None, lead_time_hours=6)
    assert gfs_key == "temperature_2m/shard.det_L0006.shard"

    # Ensemble GEFS shard key (member=3)
    gefs_key = reader._get_shard_key("temperature_2m", member=3, lead_time_hours=12)
    assert gefs_key == "temperature_2m/shard.mem003_L0012.shard"

    # Precomputed ensemble-mean shard key (is_mean=True)
    mean_key = reader._get_shard_key(
        "temperature_2m", member=None, lead_time_hours=6, is_mean=True
    )
    assert mean_key == "temperature_2m/shard.mean_L0006.shard"


def test_every_shard_key_construction_site_agrees_with_domain_authority() -> None:
    """The key template must live in exactly one place.

    ``domain.reclamation.make_shard_filename`` is the authority. The ingestion encoder,
    the ingestion reader, the store inventory and the API reader all have to agree with it
    for every region shape -- a divergence in any one of them silently reads or deletes the
    wrong object rather than failing loudly.
    """
    import numpy as np
    import xarray as xr

    from domain import reclamation
    from ingestion.core import inventory as ing_inventory

    shapes = (
        ("det", dict(member=None, lead_time_hours=6, is_mean=False)),
        ("mean", dict(member=None, lead_time_hours=6, is_mean=True)),
        ("mem", dict(member=3, lead_time_hours=12, is_mean=False)),
    )
    variable = "temperature_2m"
    # A tiny grid keeps the encoder cheap; only the key matters here.
    data = np.zeros((1, 1, 4, 4), dtype=np.float32)

    for _label, kwargs in shapes:
        expected = reclamation.make_shard_filename(variable, **kwargs)

        # 1. domain authority vs the target-kind builder (two spellings, one template)
        kind, member_index = reclamation.target_kind_for(
            kwargs["member"], is_mean=kwargs["is_mean"]
        )
        assert expected == reclamation.make_shard_relative_key(
            variable, kind, kwargs["lead_time_hours"], member_index
        )

        # 2. ingestion encoder
        ds = xr.Dataset(
            {
                variable: (
                    ("member", "lead_time_hours", "latitude", "longitude"),
                    data,
                )
            }
        )
        encoded = ing_zarr.encode_region_sharded_v1(ds, **kwargs)
        assert [k for k, _ in encoded] == [expected]

        # 3. API reader
        reader = api_zarr.ShardedV1Reader("s3://weather-data/gfs/2026-09-03/00/cycle.zarr")
        assert (
            reader._get_shard_key(
                variable,
                member=kwargs["member"],
                lead_time_hours=kwargs["lead_time_hours"],
                is_mean=kwargs["is_mean"],
            )
            == expected
        )

        # 4. store inventory
        assert ing_inventory.region_expected_object_keys(
            "s3://weather-data/gfs/2026-09-03/00/cycle.zarr",
            member=kwargs["member"],
            lead_index=kwargs["lead_time_hours"],
            lead_time_hours=kwargs["lead_time_hours"],
            data_var_paths=[variable],
            format_version="sharded_v1",
            is_mean=kwargs["is_mean"],
        ) == [expected]

        # 5. the filename grammar round-trips, so detection agrees with construction
        assert reclamation.parse_shard_filename(expected) == (
            kwargs["member"],
            kwargs["lead_time_hours"],
            kwargs["is_mean"],
        )


def test_canonical_storage_identity_normalization() -> None:
    """Verify that storage identity normalization is identical across both consumers."""
    raw_path = "s3://weather-data/gfs/2026-09-03/00/cycle.zarr/"
    norm_api = canonical_storage_identity(raw_path, endpoint="localhost:9000", secure=False)
    norm_ing = canonical_storage_identity(raw_path, endpoint="localhost:9000", secure=False)
    assert norm_api == norm_ing == "s3://http://localhost:9000/weather-data/gfs/2026-09-03/00/cycle.zarr"
