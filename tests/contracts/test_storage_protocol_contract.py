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


def test_aggregate_reader_agrees_with_the_ingestion_writer(tmp_path) -> None:
    """The API reader and the ingestion writer must agree on the aggregate container.

    This is the cross-service assertion, and it belongs here rather than in the API's own
    suite: the API tier is independently deployable and must not import the ingestion package
    (``docs/ARCHITECTURE.md`` 3.1/3.5), so only this suite can exercise both sides at once.
    A geometry, key or ordering mismatch between them would otherwise surface as misaligned
    statistic planes rather than as an error.
    """
    import numpy as np
    import os

    from api.core.aggregate_reader import (
        AggregateShardReader,
        aggregate_shard_key,
        recover_geometry,
    )
    from domain.aggregate import KIND_QUANTILE_FUNCTION, AggregateSpec, compute_aggregate
    from ingestion.core.aggregate_writer import (
        encode_aggregate_shard,
        layout_for_spec,
    )

    grid_lat, grid_lon = 128, 160
    variable, lead = "temperature_2m", 6
    members = np.random.default_rng(0).normal(280.0, 8.0, (5, grid_lat, grid_lon))
    store = str(tmp_path / "cycle.zarr")

    for spec in (
        AggregateSpec(n_bins=4),
        AggregateSpec(kind=KIND_QUANTILE_FUNCTION),
    ):
        fields = compute_aggregate(members.astype(np.float32), spec)
        layout = layout_for_spec(spec, grid_lat=grid_lat, grid_lon=grid_lon)
        container = encode_aggregate_shard(
            list(fields), layout, member_count=members.shape[0]
        )
        key = aggregate_shard_key(variable, lead)
        full = os.path.join(store, *key.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(container)

        # 1. the reader recovers exactly the writer's geometry
        reader = AggregateShardReader(store)
        geometry = reader.open(variable, lead)
        assert geometry is not None, spec.kind
        assert geometry.n_fields == spec.n_fields
        assert geometry.num_chunks == layout.num_chunks
        assert geometry.lat_chunks == layout.lat_chunks
        assert geometry.lon_chunks == layout.lon_chunks
        assert recover_geometry(layout.to_descriptor()) == geometry

        # 1b. and the member count the writer recorded, which an aggregate cannot derive from
        # the statistics it stores
        assert reader.member_count(variable, lead, observed=True) == members.shape[0]
        assert reader.member_count(variable, lead, observed=False) == 0

        # 2. and decodes the fields the writer was given, chunk for chunk
        for row in range(geometry.lat_chunks):
            for col in range(geometry.lon_chunks):
                stack = reader.read_location(
                    variable, lead_time_hours=lead, chunk_row=row, chunk_col=col
                )
                assert stack is not None, (spec.kind, row, col)
                r0, c0 = row * layout.chunk_lat, col * layout.chunk_lon
                r1 = min(r0 + layout.chunk_lat, grid_lat)
                c1 = min(c0 + layout.chunk_lon, grid_lon)
                for field in range(spec.n_fields):
                    assert np.array_equal(
                        stack[field][: r1 - r0, : c1 - c0],
                        fields[field][r0:r1, c0:c1],
                    ), (spec.kind, row, col, field)
