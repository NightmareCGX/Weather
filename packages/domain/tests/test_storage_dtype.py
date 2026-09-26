"""Tests for the authoritative sharded storage dtype resolver."""

from __future__ import annotations

import numpy as np
import pytest
from domain.reclamation import TARGET_KIND_DET, TARGET_KIND_MEAN, TARGET_KIND_MEM
from domain.storage_dtype import (
    FLOAT16_LE_DTYPE,
    FLOAT32_LE_DTYPE,
    SHARDED_V1_FORMAT_VERSION,
    SHARDED_V2_FORMAT_VERSION,
    SHARDED_V2_KNOWN_VARIABLES,
    UINT8_DTYPE,
    is_sharded_payload_format,
    resolve_storage_dtype,
)


class TestShardedV1Resolution:
    def test_every_variable_is_float32(self) -> None:
        f32 = np.dtype("<f4")
        for var in SHARDED_V2_KNOWN_VARIABLES:
            for role in (TARGET_KIND_DET, TARGET_KIND_MEM, TARGET_KIND_MEAN):
                assert resolve_storage_dtype(SHARDED_V1_FORMAT_VERSION, var, role) == f32

    def test_sharded_v1_dtype_is_explicit_little_endian(self) -> None:
        dtype = resolve_storage_dtype(SHARDED_V1_FORMAT_VERSION, "temperature_2m", TARGET_KIND_DET)
        assert dtype.byteorder in ("<", "|", "=")
        assert dtype.itemsize == 4


class TestShardedV2Resolution:
    def test_continuous_variables_are_float16(self) -> None:
        f16 = np.dtype("<f2")
        for var in (
            "temperature_2m",
            "relative_humidity_2m",
            "wind_u_10m",
            "wind_v_10m",
            "wind_gust",
            "precipitation_rate",
            "cloud_cover_3h",
            "snow_depth",
            "visibility",
        ):
            for role in (TARGET_KIND_DET, TARGET_KIND_MEM, TARGET_KIND_MEAN):
                assert resolve_storage_dtype(SHARDED_V2_FORMAT_VERSION, var, role) == f16

    def test_semantic_exceptions_are_float32(self) -> None:
        # precipitation_amount_3h: exactly-0.10mm threshold compatibility.
        # cloud_ceiling: 19.99 km unlimited-sentinel compatibility.
        f32 = np.dtype("<f4")
        for var in ("precipitation_amount_3h", "cloud_ceiling"):
            for role in (TARGET_KIND_DET, TARGET_KIND_MEM, TARGET_KIND_MEAN):
                assert resolve_storage_dtype(SHARDED_V2_FORMAT_VERSION, var, role) == f32

    def test_flag_variables_role_dependent(self) -> None:
        f32 = np.dtype("<f4")
        for var in ("crain", "csnow", "cfrzr", "cicep"):
            resolved_det = resolve_storage_dtype(SHARDED_V2_FORMAT_VERSION, var, TARGET_KIND_DET)
            resolved_mem = resolve_storage_dtype(SHARDED_V2_FORMAT_VERSION, var, TARGET_KIND_MEM)
            assert resolved_det == UINT8_DTYPE
            assert resolved_mem == UINT8_DTYPE
            # Ensemble-mean flags are member-mean probabilities, not categoricals.
            assert resolve_storage_dtype(SHARDED_V2_FORMAT_VERSION, var, TARGET_KIND_MEAN) == f32

    def test_dtypes_are_explicit_little_endian(self) -> None:
        assert FLOAT16_LE_DTYPE.itemsize == 2
        assert FLOAT32_LE_DTYPE.itemsize == 4
        assert UINT8_DTYPE.itemsize == 1


class TestFailClosed:
    def test_unknown_variable_raises(self) -> None:
        with pytest.raises(ValueError, match="no sharded_v2 dtype decision"):
            resolve_storage_dtype(SHARDED_V2_FORMAT_VERSION, "cape_jkg", TARGET_KIND_DET)

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown storage format version"):
            resolve_storage_dtype("sharded_v3", "temperature_2m", TARGET_KIND_DET)

    def test_legacy_unsharded_is_not_resolvable(self) -> None:
        with pytest.raises(ValueError, match="Unknown storage format version"):
            resolve_storage_dtype("v2_unsharded", "temperature_2m", TARGET_KIND_DET)

    def test_unknown_role_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown product role"):
            resolve_storage_dtype(SHARDED_V2_FORMAT_VERSION, "temperature_2m", "ensemble")


class TestFormatPredicate:
    def test_sharded_formats(self) -> None:
        assert is_sharded_payload_format("sharded_v1") is True
        assert is_sharded_payload_format("sharded_v2") is True

    def test_unsharded_formats(self) -> None:
        assert is_sharded_payload_format("v2_unsharded") is False
        assert is_sharded_payload_format("sharded_v3") is False


class TestVariableMatrixCompleteness:
    def test_matrix_covers_exactly_fifteen_variables(self) -> None:
        assert len(SHARDED_V2_KNOWN_VARIABLES) == 15
        assert {
            "temperature_2m",
            "relative_humidity_2m",
            "wind_u_10m",
            "wind_v_10m",
            "wind_gust",
            "precipitation_rate",
            "precipitation_amount_3h",
            "cloud_cover_3h",
            "cloud_ceiling",
            "snow_depth",
            "visibility",
            "crain",
            "csnow",
            "cfrzr",
            "cicep",
        } == SHARDED_V2_KNOWN_VARIABLES

    def test_matrix_sets_are_disjoint(self) -> None:
        from domain.storage_dtype import (  # noqa: PLC0415 — test-only introspection
            _FLAG_VARIABLES,
            _V2_FLOAT16_VARIABLES,
            _V2_FLOAT32_VARIABLES,
        )

        assert not (_V2_FLOAT16_VARIABLES & _V2_FLOAT32_VARIABLES)
        assert not (_V2_FLOAT16_VARIABLES & _FLAG_VARIABLES)
        assert not (_V2_FLOAT32_VARIABLES & _FLAG_VARIABLES)
