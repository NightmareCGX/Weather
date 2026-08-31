"""Unit tests for pure meteorological domain cloud functions and models."""

import math

import numpy as np
import pytest
from domain.models.cloud import (
    classify_cloud_ceiling,
    cloud_ceiling_ensemble_summary,
    cloud_cover_ensemble_summary,
    compute_low_ceiling_probability,
    reconstruct_cloud_cover_3h,
)


class TestCloudCoverReconstruction:
    """Test suite for 3-hour cloud cover interval-average reconstruction."""

    def test_reconstruct_scalar_valid_range(self) -> None:
        # Standard unclipped reconstruction: x = 2 * 50 - 40 = 60
        res = reconstruct_cloud_cover_3h(50.0, 40.0)
        assert res == 60.0

        # Exact 0 boundary: x = 2 * 20 - 40 = 0
        res_0 = reconstruct_cloud_cover_3h(20.0, 40.0)
        assert res_0 == 0.0

        # Exact 100 boundary: x = 2 * 75 - 50 = 100
        res_100 = reconstruct_cloud_cover_3h(75.0, 50.0)
        assert res_100 == 100.0

    def test_reconstruct_scalar_minor_undershoot_clipping(self) -> None:
        # Undershoot inside [-5, 0): x = 2 * 18 - 40 = -4.0 -> clipped to 0.0
        res = reconstruct_cloud_cover_3h(18.0, 40.0)
        assert res == 0.0

        # Exact boundary -5.0 -> clipped to 0.0
        res_edge = reconstruct_cloud_cover_3h(17.5, 40.0)
        assert res_edge == 0.0

    def test_reconstruct_scalar_minor_overshoot_clipping(self) -> None:
        # Overshoot inside (100, 105]: x = 2 * 76 - 50 = 102.0 -> clipped to 100.0
        res = reconstruct_cloud_cover_3h(76.0, 50.0)
        assert res == 100.0

        # Exact boundary 105.0 -> clipped to 100.0
        res_edge = reconstruct_cloud_cover_3h(77.5, 50.0)
        assert res_edge == 100.0

    def test_reconstruct_scalar_invalid_guardrail_failure(self) -> None:
        # Gross undershoot < -5.0: x = 2 * 10 - 40 = -20.0 -> NaN
        res_low = reconstruct_cloud_cover_3h(10.0, 40.0)
        assert math.isnan(res_low)

        # Gross overshoot > 105.0: x = 2 * 80 - 50 = 110.0 -> NaN
        res_high = reconstruct_cloud_cover_3h(80.0, 50.0)
        assert math.isnan(res_high)

    def test_reconstruct_scalar_nan_inputs(self) -> None:
        assert math.isnan(reconstruct_cloud_cover_3h(float("nan"), 50.0))
        assert math.isnan(reconstruct_cloud_cover_3h(50.0, float("nan")))
        assert math.isnan(reconstruct_cloud_cover_3h(float("nan"), float("nan")))

    def test_reconstruct_array_vectorized(self) -> None:
        c6 = np.array([50.0, 20.0, 18.0, 17.5, 76.0, 77.5, 10.0, 80.0, np.nan], dtype=np.float64)
        c3 = np.array([40.0, 40.0, 40.0, 40.0, 50.0, 50.0, 40.0, 50.0, 50.0], dtype=np.float64)

        result: np.ndarray = reconstruct_cloud_cover_3h(c6, c3)
        expected = np.array(
            [60.0, 0.0, 0.0, 0.0, 100.0, 100.0, np.nan, np.nan, np.nan], dtype=np.float64
        )

        np.testing.assert_allclose(result[:6], expected[:6], rtol=1e-5)
        assert np.isnan(result[6:]).all()

    def test_reconstruct_array_float32_preservation(self) -> None:
        c6 = np.array([50.0, 18.0], dtype=np.float32)
        c3 = np.array([40.0, 40.0], dtype=np.float32)
        result: np.ndarray = reconstruct_cloud_cover_3h(c6, c3)
        assert result.dtype == np.float32
        assert result[0] == 60.0
        assert result[1] == 0.0


class TestCloudCeilingClassification:
    """Test suite for cloud ceiling height and unlimited sentinel classification."""

    def test_classify_finite_height(self) -> None:
        cls = classify_cloud_ceiling(1200.0)
        assert not cls.is_unlimited
        assert cls.height_m == 1200.0

    def test_classify_zero_height(self) -> None:
        cls = classify_cloud_ceiling(0.0)
        assert not cls.is_unlimited
        assert cls.height_m == 0.0

    def test_classify_unlimited_sentinel(self) -> None:
        cls_sentinel = classify_cloud_ceiling(20000.0)
        assert cls_sentinel.is_unlimited
        assert cls_sentinel.height_m is None

        # Threshold boundary exactly 19990.0
        cls_edge = classify_cloud_ceiling(19990.0)
        assert cls_edge.is_unlimited
        assert cls_edge.height_m is None

        # Just below threshold 19989.0 is treated as high finite ceiling
        cls_below = classify_cloud_ceiling(19989.0)
        assert not cls_below.is_unlimited
        assert cls_below.height_m == 19989.0

    def test_classify_none_and_nan(self) -> None:
        cls_none = classify_cloud_ceiling(None)
        assert not cls_none.is_unlimited
        assert cls_none.height_m is None

        cls_nan = classify_cloud_ceiling(float("nan"))
        assert not cls_nan.is_unlimited
        assert cls_nan.height_m is None


class TestCloudEnsembleSummaries:
    """Test suite for GEFS cloud cover and ceiling ensemble summary calculations."""

    def test_cloud_cover_ensemble_full_valid(self) -> None:
        members = [50.0 + i for i in range(30)]  # 50 to 79
        summary = cloud_cover_ensemble_summary(members)
        assert summary is not None
        assert summary.valid_member_count == 30
        assert summary.invalid_member_count == 0
        assert summary.mean == pytest.approx(64.5, abs=1e-2)
        assert summary.median == pytest.approx(64.5, abs=1e-2)
        assert "p10" in summary.percentiles
        assert "p50" in summary.percentiles
        assert "p90" in summary.percentiles

    def test_cloud_cover_ensemble_partial_valid_above_gate(self) -> None:
        # 25 valid members and 5 invalid (NaN, out-of-bounds)
        members = [50.0] * 25 + [float("nan"), -10.0, 110.0, None, float("nan")]
        summary = cloud_cover_ensemble_summary(members)
        assert summary is not None
        assert summary.valid_member_count == 25
        assert summary.invalid_member_count == 5
        assert summary.mean == 50.0
        assert summary.median == 50.0

    def test_cloud_cover_ensemble_below_gate_suppression(self) -> None:
        # 20 valid members and 10 invalid -> N_valid = 20 < 21 -> returns None
        members = [50.0] * 20 + [float("nan")] * 10
        summary = cloud_cover_ensemble_summary(members)
        assert summary is None

    def test_cloud_ceiling_ensemble_mixed_state_robust(self) -> None:
        # 18 finite members (1000m) and 12 unlimited members (20000m)
        members = [1000.0 + (i * 10) for i in range(18)] + [20000.0] * 12
        summary = cloud_ceiling_ensemble_summary(members)
        assert summary is not None
        assert summary.valid_member_count == 30
        assert summary.finite_member_count == 18
        assert summary.unlimited_member_count == 12
        assert summary.unlimited_probability == pytest.approx(12.0 / 30.0, abs=1e-4)

        # Finite count 18 >= 10 -> conditional percentiles are computed
        assert summary.conditional_median_m is not None
        assert summary.conditional_percentiles_m is not None
        assert summary.conditional_mean_m == pytest.approx(1085.0, abs=1e-1)
        assert summary.conditional_percentiles_m["p50"] == pytest.approx(1085.0, abs=1e-1)

    def test_cloud_ceiling_ensemble_high_unlimited_suppression(self) -> None:
        # 4 finite members and 26 unlimited members
        members = [1000.0, 1200.0, 1500.0, 1800.0] + [20000.0] * 26
        summary = cloud_ceiling_ensemble_summary(members)
        assert summary is not None
        assert summary.valid_member_count == 30
        assert summary.finite_member_count == 4
        assert summary.unlimited_member_count == 26
        assert summary.unlimited_probability == pytest.approx(26.0 / 30.0, abs=1e-4)

        # Finite count 4 < 10 -> conditional percentiles suppressed
        assert summary.conditional_median_m is None
        assert summary.conditional_mean_m is None
        assert summary.conditional_percentiles_m is None

    def test_cloud_ceiling_ensemble_all_unlimited(self) -> None:
        members = [20000.0] * 30
        summary = cloud_ceiling_ensemble_summary(members)
        assert summary is not None
        assert summary.valid_member_count == 30
        assert summary.finite_member_count == 0
        assert summary.unlimited_member_count == 30
        assert summary.unlimited_probability == 1.0
        assert summary.conditional_percentiles_m is None

    def test_cloud_ceiling_ensemble_below_validity_gate(self) -> None:
        # 15 valid members and 15 NaNs -> N_valid = 15 < 21 -> returns None
        members = [1000.0] * 10 + [20000.0] * 5 + [float("nan")] * 15
        summary = cloud_ceiling_ensemble_summary(members)
        assert summary is None

    def test_compute_low_ceiling_probability(self) -> None:
        # 10 members at 500m, 10 members at 1500m, 10 members unlimited (20000m)
        members = [500.0] * 10 + [1500.0] * 10 + [20000.0] * 10
        # P(Ceiling <= 1000m) should be 10 / 30 = 0.3333
        prob_1000 = compute_low_ceiling_probability(members, 1000.0)
        assert prob_1000 == pytest.approx(10.0 / 30.0, abs=1e-4)

        # P(Ceiling <= 2000m) should be 20 / 30 = 0.6667
        prob_2000 = compute_low_ceiling_probability(members, 2000.0)
        assert prob_2000 == pytest.approx(20.0 / 30.0, abs=1e-4)

        # Below valid gate (<21) -> returns None
        prob_invalid = compute_low_ceiling_probability(
            [500.0] * 10 + [float("nan")] * 20, 1000.0
        )
        assert prob_invalid is None
