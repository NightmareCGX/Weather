"""Tests for the sharded_v2 storage format: byte layout, writer guards, cycle
format freeze, rollback, and the two f32 semantic-compatibility regressions.

Contract under test (frozen):
* continuous fields -> little-endian ``<f2`` (2 bytes/value);
* ``precipitation_amount_3h`` / ``cloud_ceiling`` -> little-endian ``<f4``
  (semantic compatibility: exactly-0.10 mm threshold, 19.99 km sentinel);
* det/member categorical flags -> ``u1`` (1 byte/value), values in {0, 1};
* ensemble-mean flags -> little-endian ``<f4`` probabilities in [0, 1];
* index/trailer/object keys identical to sharded_v1;
* ``.zarray`` metadata equals the actual payload dtype (v2 only).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numcodecs import Zstd

from domain.storage_dtype import (
    SHARDED_V1_FORMAT_VERSION,
    SHARDED_V2_FORMAT_VERSION,
    resolve_storage_dtype,
)

from ingestion.core.base import (
    CategoricalDomainViolationError,
    CycleFormatConflictError,
    F16RangeViolationError,
)
from ingestion.core.config import settings
from ingestion.core.zarr_writer import (
    INDEX_ENTRY_SIZE,
    SHARD_MAGIC,
    TRAILER_SIZE,
    assert_cycle_format_allowed,
    encode_region_sharded,
    encode_region_sharded_v2,
    parse_sharded_v1_index,
)

COMPRESSOR = Zstd(level=5)


def _v2_region_dataset(
    lead: int = 6,
    member: int | None = 1,
    *,
    temperature: float = 21.5,
    precip: float | None = 0.1,
    ceiling: float | None = 19.991,
    flag: float = 1.0,
    cloud: float = 87.5,
) -> xr.Dataset:
    lat = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    lon = np.linspace(0.0, 359.75, 1440, dtype=np.float32)
    dims = ("latitude", "longitude")
    shape = (721, 1440)
    data_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {
        "temperature_2m": (dims, np.full(shape, temperature, dtype=np.float32)),
        "cloud_cover_3h": (dims, np.full(shape, cloud, dtype=np.float32)),
    }
    if precip is not None:
        data_vars["precipitation_amount_3h"] = (
            dims,
            np.full(shape, precip, dtype=np.float32),
        )
    if ceiling is not None:
        data_vars["cloud_ceiling"] = (dims, np.full(shape, ceiling, dtype=np.float32))
    data_vars["crain"] = (dims, np.full(shape, flag, dtype=np.float32))
    coords = {
        "lead_time_hours": [lead],
        "latitude": lat,
        "longitude": lon,
    }
    if member is not None:
        coords["member"] = [member]
    return xr.Dataset(data_vars=data_vars, coords=coords)


def _decoded_chunks(payload: bytes) -> list[np.ndarray]:
    """Decode every chunk of a shard container back to raw arrays."""
    num_chunks, index_size, magic = struct.unpack("<III", payload[-TRAILER_SIZE:])
    assert magic == SHARD_MAGIC
    index_bytes = payload[-(TRAILER_SIZE + index_size) : -TRAILER_SIZE]
    entries = parse_sharded_v1_index(index_bytes, num_chunks)
    out = []
    for off, length in entries:
        assert length > 0
        out.append(np.frombuffer(COMPRESSOR.decode(payload[off : off + length]), dtype=np.uint8))
    return out


class TestV2ByteLayout:
    def test_f16_fields_are_two_bytes_per_value(self) -> None:
        ds = _v2_region_dataset()
        shards = dict(
            encode_region_sharded_v2(ds, member=1, lead_time_hours=6)
        )
        payload = shards["temperature_2m/shard.mem001_L0006.shard"]
        num_chunks, index_size, magic = struct.unpack("<III", payload[-TRAILER_SIZE:])
        assert magic == SHARD_MAGIC and num_chunks == 120
        index_bytes = payload[-(TRAILER_SIZE + index_size) : -TRAILER_SIZE]
        entries = parse_sharded_v1_index(index_bytes, num_chunks)
        first = COMPRESSOR.decode(payload[entries[0][0] : entries[0][0] + entries[0][1]])
        assert len(first) == 100 * 100 * 2  # f16: 2 bytes/value

    def test_f32_exceptions_are_four_bytes_per_value(self) -> None:
        ds = _v2_region_dataset()
        shards = dict(encode_region_sharded_v2(ds, member=1, lead_time_hours=6))
        for var in ("precipitation_amount_3h", "cloud_ceiling"):
            payload = shards[f"{var}/shard.mem001_L0006.shard"]
            index_bytes = payload[-(TRAILER_SIZE + 120 * INDEX_ENTRY_SIZE) : -TRAILER_SIZE]
            off, length = parse_sharded_v1_index(index_bytes, 120)[0]
            first = COMPRESSOR.decode(payload[off : off + length])
            assert len(first) == 100 * 100 * 4  # f32: 4 bytes/value

    def test_det_flags_are_one_byte_per_value(self) -> None:
        ds = _v2_region_dataset()
        shards = dict(encode_region_sharded_v2(ds, member=1, lead_time_hours=6))
        payload = shards["crain/shard.mem001_L0006.shard"]
        index_bytes = payload[-(TRAILER_SIZE + 120 * INDEX_ENTRY_SIZE) : -TRAILER_SIZE]
        off, length = parse_sharded_v1_index(index_bytes, 120)[0]
        first = COMPRESSOR.decode(payload[off : off + length])
        assert len(first) == 100 * 100  # u1: 1 byte/value
        assert np.frombuffer(first, dtype=np.uint8)[0] == 1

    def test_mean_flags_are_f32_probabilities(self) -> None:
        ds = _v2_region_dataset(member=None, flag=0.65)
        shards = dict(
            encode_region_sharded_v2(ds, member=None, lead_time_hours=6, is_mean=True)
        )
        payload = shards["crain/shard.mean_L0006.shard"]
        index_bytes = payload[-(TRAILER_SIZE + 120 * INDEX_ENTRY_SIZE) : -TRAILER_SIZE]
        off, length = parse_sharded_v1_index(index_bytes, 120)[0]
        first = COMPRESSOR.decode(payload[off : off + length])
        assert len(first) == 100 * 100 * 4
        values = np.frombuffer(first, dtype="<f4")
        assert values[0] == np.float32(0.65)

    def test_explicit_little_endian_f16_bytes(self) -> None:
        """f16 payload bytes must be the little-endian encoding of the value."""
        ds = _v2_region_dataset(temperature=290.0)
        shards = dict(encode_region_sharded_v2(ds, member=None, lead_time_hours=6))
        payload = shards["temperature_2m/shard.det_L0006.shard"]
        index_bytes = payload[-(TRAILER_SIZE + 120 * INDEX_ENTRY_SIZE) : -TRAILER_SIZE]
        off, length = parse_sharded_v1_index(index_bytes, 120)[0]
        raw = COMPRESSOR.decode(payload[off : off + length])
        assert raw[:2] == struct.pack("<e", 290.0)
        assert np.frombuffer(raw[:2], dtype="<f2")[0] == np.float16(290.0)

    def test_object_keys_match_v1_layout(self) -> None:
        ds = _v2_region_dataset()
        v2_keys = {k for k, _ in encode_region_sharded_v2(ds, member=7, lead_time_hours=9)}
        v1_keys = {
            k
            for k, _ in encode_region_sharded(
                ds,
                member=7,
                lead_time_hours=9,
                format_version=SHARDED_V1_FORMAT_VERSION,
            )
        }
        assert v2_keys == v1_keys

    def test_dispatch_wrapper_routes_v1_to_frozen_encoder(self) -> None:
        ds = _v2_region_dataset()
        v1 = dict(
            encode_region_sharded(
                ds, member=1, lead_time_hours=6, format_version=SHARDED_V1_FORMAT_VERSION
            )
        )
        payload = v1["temperature_2m/shard.mem001_L0006.shard"]
        index_bytes = payload[-(TRAILER_SIZE + 120 * INDEX_ENTRY_SIZE) : -TRAILER_SIZE]
        off, length = parse_sharded_v1_index(index_bytes, 120)[0]
        first = COMPRESSOR.decode(payload[off : off + length])
        assert len(first) == 100 * 100 * 4  # v1 frozen: float32


class TestV2WriterGuards:
    def test_nan_allowed_and_does_not_raise(self) -> None:
        ds = _v2_region_dataset(temperature=21.5)
        ds["temperature_2m"][0, 0] = np.nan
        shards = dict(encode_region_sharded_v2(ds, member=1, lead_time_hours=6))
        assert "temperature_2m/shard.mem001_L0006.shard" in shards

    def test_inf_rejected_before_cast(self) -> None:
        ds = _v2_region_dataset(temperature=21.5)
        ds["temperature_2m"][0, 0] = np.inf
        with pytest.raises(F16RangeViolationError) as exc:
            encode_region_sharded_v2(ds, member=1, lead_time_hours=6)
        message = str(exc.value)
        assert "variable='temperature_2m'" in message
        assert "product_role='mem'" in message
        assert "shard_key='temperature_2m/shard.mem001_L0006.shard'" in message
        assert "inf=1" in message

    def test_overflow_rejected(self) -> None:
        ds = _v2_region_dataset(temperature=70000.0)
        with pytest.raises(F16RangeViolationError) as exc:
            encode_region_sharded_v2(ds, member=1, lead_time_hours=6)
        assert "overflow=1" in str(exc.value)

    def test_max_finite_f16_allowed(self) -> None:
        ds = _v2_region_dataset(temperature=65504.0)
        shards = dict(encode_region_sharded_v2(ds, member=1, lead_time_hours=6))
        assert "temperature_2m/shard.mem001_L0006.shard" in shards

    def test_f32_exceptions_do_not_get_f16_guard(self) -> None:
        # The guard is driven by the resolved storage dtype, not the variable
        # class: precipitation/ceiling keep <f4 and are never f16-guarded.
        ds = _v2_region_dataset(precip=1e6, ceiling=1e6)
        shards = dict(encode_region_sharded_v2(ds, member=1, lead_time_hours=6))
        assert "precipitation_amount_3h/shard.mem001_L0006.shard" in shards
        assert "cloud_ceiling/shard.mem001_L0006.shard" in shards

    def test_det_flag_domain_violation_rejected(self) -> None:
        ds = _v2_region_dataset(flag=2.0)
        with pytest.raises(CategoricalDomainViolationError):
            encode_region_sharded_v2(ds, member=1, lead_time_hours=6)

    def test_mean_flag_domain_violation_rejected(self) -> None:
        ds = _v2_region_dataset(member=None, flag=1.5)
        with pytest.raises(CategoricalDomainViolationError):
            encode_region_sharded_v2(ds, member=None, lead_time_hours=6, is_mean=True)


class TestCycleFormatFreeze:
    @pytest.fixture(autouse=True)
    def _reset_snapshots(self) -> None:
        from ingestion.core import zarr_writer

        zarr_writer._store_format_snapshots.clear()
        yield
        zarr_writer._store_format_snapshots.clear()

    def test_first_commit_snapshots_then_conflict(self, tmp_path: Path) -> None:
        store = str(tmp_path / "cycle.zarr")
        assert_cycle_format_allowed(store, SHARDED_V2_FORMAT_VERSION)
        with pytest.raises(CycleFormatConflictError, match="sharded_v1"):
            assert_cycle_format_allowed(store, SHARDED_V1_FORMAT_VERSION)
        # Same format again is always fine.
        assert_cycle_format_allowed(store, SHARDED_V2_FORMAT_VERSION)

    def test_writer_rejects_mid_cycle_format_flip(self, tmp_path: Path, monkeypatch) -> None:
        """commit_region under v2 then v1 on the same store fails loudly."""
        from ingestion.core.zarr_writer import commit_region, prepare_run_store

        store = str(tmp_path / "flip.zarr")
        monkeypatch.setattr(settings, "STORAGE_FORMAT_VERSION", SHARDED_V2_FORMAT_VERSION)
        ds = _v2_region_dataset(lead=0)
        prepare_run_store(ds, store, expected_lead_time_hours=(0, 6), expected_members=(1,))
        commit_region(ds, store, lead_time_hours=0, member=1)

        monkeypatch.setattr(settings, "STORAGE_FORMAT_VERSION", SHARDED_V1_FORMAT_VERSION)
        ds6 = _v2_region_dataset(lead=6)
        with pytest.raises(CycleFormatConflictError):
            commit_region(ds6, store, lead_time_hours=6, member=1)


class TestV2StoreRoundtrip:
    def test_zarray_metadata_matches_payload_dtype(self, tmp_path: Path, monkeypatch) -> None:
        """v2 only: .zarray dtype must equal the resolver's payload dtype."""
        from ingestion.core.zarr_writer import prepare_run_store

        monkeypatch.setattr(settings, "STORAGE_FORMAT_VERSION", SHARDED_V2_FORMAT_VERSION)
        store = str(tmp_path / "v2.zarr")
        prepare_run_store(
            _v2_region_dataset(lead=0),
            store,
            expected_lead_time_hours=(0, 6),
            expected_members=(1,),
        )
        expected = {
            "temperature_2m": "<f2",
            "cloud_cover_3h": "<f2",
            "precipitation_amount_3h": "<f4",
            "cloud_ceiling": "<f4",
            # zarr serializes 1-byte dtypes with the "|" (not-applicable) order.
            "crain": "|u1",
        }
        for var, dtype_str in expected.items():
            zarray = json.loads((Path(store) / var / ".zarray").read_text())
            assert zarray["dtype"] == dtype_str, var

    def test_v1_zarray_behavior_unchanged(self, tmp_path: Path, monkeypatch) -> None:
        """v1 keeps its historical metadata behavior (seed dataset dtype)."""
        from ingestion.core.zarr_writer import prepare_run_store

        monkeypatch.setattr(settings, "STORAGE_FORMAT_VERSION", SHARDED_V1_FORMAT_VERSION)
        store = str(tmp_path / "v1.zarr")
        prepare_run_store(
            _v2_region_dataset(lead=0),
            store,
            expected_lead_time_hours=(0, 6),
            expected_members=(1,),
        )
        zarray = json.loads((Path(store) / "crain" / ".zarray").read_text())
        # Seed dataset carries float32 flags (pre-P2-fix decode), so v1
        # metadata stays float32 — frozen historical behavior.
        assert zarray["dtype"] == "<f4"

    def test_commit_and_read_slice_native_dtype(self, tmp_path: Path, monkeypatch) -> None:
        from ingestion.core.zarr_writer import commit_region, prepare_run_store, read_slice

        monkeypatch.setattr(settings, "STORAGE_FORMAT_VERSION", SHARDED_V2_FORMAT_VERSION)
        store = str(tmp_path / "roundtrip.zarr")
        prepare_run_store(
            _v2_region_dataset(lead=0),
            store,
            expected_lead_time_hours=(0, 6),
            expected_members=(1,),
        )
        commit_region(_v2_region_dataset(lead=0), store, lead_time_hours=0, member=1)

        temp = read_slice(store, "temperature_2m", lead_time_hours=0, member=1)
        assert temp is not None and temp.dtype == np.dtype("<f2")
        assert temp[0, 0] == np.float16(21.5)

        precip = read_slice(store, "precipitation_amount_3h", lead_time_hours=0, member=1)
        assert precip is not None and precip.dtype == np.dtype("<f4")
        assert precip[0, 0] == np.float32(0.1)

        ceiling = read_slice(store, "cloud_ceiling", lead_time_hours=0, member=1)
        assert ceiling is not None and ceiling.dtype == np.dtype("<f4")
        assert ceiling[0, 0] == np.float32(19.991)

        crain = read_slice(store, "crain", lead_time_hours=0, member=1)
        assert crain is not None and crain.dtype == np.dtype("uint8")
        assert crain[0, 0] == 1

        cloud = read_slice(store, "cloud_cover_3h", lead_time_hours=0, member=1)
        assert cloud is not None and cloud.dtype == np.dtype("<f2")
        assert cloud[0, 0] == np.float16(87.5)

    def test_v1_read_slice_still_float32(self, tmp_path: Path, monkeypatch) -> None:
        from ingestion.core.zarr_writer import commit_region, prepare_run_store, read_slice

        monkeypatch.setattr(settings, "STORAGE_FORMAT_VERSION", SHARDED_V1_FORMAT_VERSION)
        store = str(tmp_path / "v1rt.zarr")
        prepare_run_store(
            _v2_region_dataset(lead=0),
            store,
            expected_lead_time_hours=(0, 6),
            expected_members=(1,),
        )
        commit_region(_v2_region_dataset(lead=0), store, lead_time_hours=0, member=1)
        temp = read_slice(store, "temperature_2m", lead_time_hours=0, member=1)
        assert temp is not None and temp.dtype == np.dtype("<f4")


class TestSemanticCompatibilityRegressions:
    def test_precipitation_exactly_0_10mm_behavior_preserved(self, tmp_path: Path) -> None:
        """np.float32(0.1) must classify identically through v1 and v2 storage.

        The production predicate promotes to float64 first
        (``float(amount) <= 0.10`` -> False -> wet); both formats store the
        exact f32 bits, so the classification must be bit-identical.
        """
        from ingestion.core.zarr_writer import commit_region, prepare_run_store, read_slice

        value = np.float32(0.1)
        for fmt in (SHARDED_V1_FORMAT_VERSION, SHARDED_V2_FORMAT_VERSION):
            store = str(tmp_path / f"precip_{fmt}.zarr")
            ds = _v2_region_dataset(lead=0, precip=float(value))
            prepare_run_store(ds, store, expected_lead_time_hours=(0,), expected_members=(1,))
            commit_region(ds, store, lead_time_hours=0, member=1)
            recovered = read_slice(store, "precipitation_amount_3h", lead_time_hours=0, member=1)
            assert recovered is not None
            assert recovered.dtype == np.dtype("<f4")
            # Bit-exact: the stored value equals the source f32(0.1) exactly.
            assert recovered[0, 0].astype(np.float32) == value
            assert float(recovered[0, 0]) == float(value)
            # Production predicate (float64 promotion): exactly-0.10 stays wet.
            assert not (float(recovered[0, 0]) <= 0.10)

    def test_cloud_ceiling_sentinel_boundaries_preserved(self, tmp_path: Path) -> None:
        """19.990..20.000 km must classify identically through v2 storage (f32 exception)."""
        from ingestion.core.zarr_writer import commit_region, prepare_run_store, read_slice

        boundary_values = [19.990, 19.991, 19.992, 19.993, 20.000]
        store = str(tmp_path / "ceiling.zarr")
        ds = _v2_region_dataset(lead=0, ceiling=boundary_values[0])
        prepare_run_store(ds, store, expected_lead_time_hours=(0,), expected_members=(1,))
        for value in boundary_values:
            commit_region(
                _v2_region_dataset(lead=0, ceiling=value), store, lead_time_hours=0, member=1
            )
            recovered = read_slice(store, "cloud_ceiling", lead_time_hours=0, member=1)
            assert recovered is not None
            assert recovered.dtype == np.dtype("<f4")
            baseline = float(np.float32(value))
            candidate = float(recovered[0, 0])
            # V2 behavior == V1 behavior, per value: (>= 19.99) -> unlimited.
            assert (candidate >= 19.99) == (baseline >= 19.99)
            assert candidate == baseline


class TestRollback:
    def test_v2_then_v1_new_cycles_stay_readable(self, tmp_path: Path, monkeypatch) -> None:
        """Rollback contract: flip writer back to v1; v2 cycles keep serving."""
        from ingestion.core.zarr_writer import commit_region, prepare_run_store, read_slice

        v2_store = str(tmp_path / "v2cycle.zarr")
        monkeypatch.setattr(settings, "STORAGE_FORMAT_VERSION", SHARDED_V2_FORMAT_VERSION)
        prepare_run_store(
            _v2_region_dataset(lead=0),
            v2_store,
            expected_lead_time_hours=(0, 6),
            expected_members=(1,),
        )
        commit_region(_v2_region_dataset(lead=0), v2_store, lead_time_hours=0, member=1)

        # Rollback: canonical writer switches back to v1 for NEW cycles.
        monkeypatch.setattr(settings, "STORAGE_FORMAT_VERSION", SHARDED_V1_FORMAT_VERSION)
        v1_store = str(tmp_path / "v1cycle.zarr")
        prepare_run_store(
            _v2_region_dataset(lead=0),
            v1_store,
            expected_lead_time_hours=(0, 6),
            expected_members=(1,),
        )
        commit_region(_v2_region_dataset(lead=0), v1_store, lead_time_hours=0, member=1)
        v1_temp = read_slice(v1_store, "temperature_2m", lead_time_hours=0, member=1)
        assert v1_temp is not None and v1_temp.dtype == np.dtype("<f4")

        # The committed v2 cycle is still served correctly by its reader path.
        v2_temp = read_slice(v2_store, "temperature_2m", lead_time_hours=0, member=1)
        assert v2_temp is not None and v2_temp.dtype == np.dtype("<f2")
        assert float(v2_temp[0, 0]) == pytest.approx(21.5, abs=0.02)


class TestResolverConsistencyWithInventory:
    def test_region_expected_object_keys_sharded_v2(self) -> None:
        from ingestion.core.inventory import region_expected_object_keys

        vars_list = [
            "temperature_2m",
            "precipitation_amount_3h",
            "cloud_ceiling",
            "crain",
            "csnow",
            "cfrzr",
            "cicep",
            "relative_humidity_2m",
            "wind_gust",
            "visibility",
            "snow_depth",
            "wind_u_10m",
            "wind_v_10m",
            "cloud_cover_3h",
        ]
        keys = region_expected_object_keys(
            "s3://weather-data/test/cycle.zarr",
            member=17,
            lead_index=2,
            lead_time_hours=6,
            data_var_paths=vars_list,
            format_version=SHARDED_V2_FORMAT_VERSION,
        )
        # Object layout identical to v1: 1 shard per variable per region.
        assert len(keys) == 14
        assert "cfrzr/shard.mem017_L0006.shard" in keys
        assert "cloud_ceiling/shard.mem017_L0006.shard" in keys


class TestResolverMatrixAgainstVariableRegistry:
    def test_ingestion_variable_registry_covered(self) -> None:
        """Every DEFAULT_VARIABLES code must have an explicit v2 dtype decision."""
        from ingestion.core.wave_runner import DEFAULT_VARIABLES

        for spec in DEFAULT_VARIABLES:
            resolved = resolve_storage_dtype(SHARDED_V2_FORMAT_VERSION, spec.code, "det")
            assert resolved.itemsize in (1, 2, 4)
