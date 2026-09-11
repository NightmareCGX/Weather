"""Unit tests for pure canonical domain logic (packages/domain/src/domain/canonical.py).

Enforces 100% test coverage across all branches, helpers, and perspectives.
"""

from __future__ import annotations

from datetime import datetime, timezone

from domain.canonical import (
    CanonicalCandidate,
    _ensure_utc,
    filter_candidates_by_physical_fence,
    select_canonical_anchor,
    select_canonical_sources_bulk,
    select_variable_source,
)


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)


def test_ensure_utc_normalizes_naive_and_aware():
    naive = datetime(2026, 9, 10, 12, 0)
    aware = _dt(2026, 9, 10, 12, 0)
    assert _ensure_utc(naive) == aware
    assert _ensure_utc(aware) == aware


def test_canonical_candidate_properties():
    c = CanonicalCandidate(
        cycle_time=_dt(2026, 9, 10, 0),
        lead_time_hours=6,
        run_id="run_1",
        store_path="s3://weather-data/gfs/cycle.zarr",
        product_types=frozenset({"surface"}),
        variables=frozenset({"temperature_2m", "wind_u_10m", "wind_v_10m"}),
    )
    assert c.valid_time == _dt(2026, 9, 10, 6)
    assert c.status == "ready"
    assert c.member_indices is None


def test_select_canonical_anchor_empty_and_non_empty():
    assert select_canonical_anchor([]) is None
    cand = CanonicalCandidate(
        cycle_time=_dt(2026, 9, 10, 0),
        lead_time_hours=6,
        run_id="run_1",
        store_path="s3://store",
        product_types=frozenset({"surface"}),
        variables=frozenset({"temperature_2m"}),
    )
    assert select_canonical_anchor([cand]) == cand


def test_select_variable_source_empty():
    assert select_variable_source([], "temperature_2m") is None


def test_select_variable_source_companion_and_fallback():
    # Candidates for valid_time = 09-10 06Z
    # cand0: cycle 09-10 06Z, lead 0 (anchor)
    # cand1: cycle 09-10 00Z, lead 6 (older cycle)
    cand0 = CanonicalCandidate(
        cycle_time=_dt(2026, 9, 10, 6),
        lead_time_hours=0,
        run_id="run_lead0",
        store_path="s3://store0",
        product_types=frozenset({"surface"}),
        variables=frozenset(
            {
                "temperature_2m",
                "precipitation_amount_3h",
                "cloud_cover_3h",
                "crain",
                "csnow",
            }
        ),
    )
    cand1 = CanonicalCandidate(
        cycle_time=_dt(2026, 9, 10, 0),
        lead_time_hours=6,
        run_id="run_lead6",
        store_path="s3://store1",
        product_types=frozenset({"surface"}),
        variables=frozenset(
            {
                "temperature_2m",
                "precipitation_amount_3h",
                "cloud_cover_3h",
                "crain",
                "csnow",
            }
        ),
    )

    cands = [cand0, cand1]

    # Ordinary variable temperature_2m uses anchor cand0
    t_src = select_variable_source(cands, "temperature_2m")
    assert t_src == cand0

    # Interval variable precipitation_amount_3h at lead 0 falls back to cand1
    p_src = select_variable_source(cands, "precipitation_amount_3h")
    assert p_src == cand1

    # Companion crain delegates to precipitation_amount_3h and gets cand1
    c_src = select_variable_source(cands, "crain")
    assert c_src == cand1

    # When anchor has lead > 0, interval variable uses anchor
    cands_pos = [cand1]
    p_src_pos = select_variable_source(cands_pos, "precipitation_amount_3h")
    assert p_src_pos == cand1

    # Cold start: only lead 0 candidate, interval variable returns None
    cands_lead0_only = [cand0]
    p_none = select_variable_source(cands_lead0_only, "precipitation_amount_3h")
    assert p_none is None

    # Positive lead but variable missing
    cand_missing_var = CanonicalCandidate(
        cycle_time=_dt(2026, 9, 10, 0),
        lead_time_hours=6,
        run_id="run_missing",
        store_path="s3://store_missing",
        product_types=frozenset({"surface"}),
        variables=frozenset({"temperature_2m"}),
    )
    assert (
        select_variable_source([cand_missing_var], "precipitation_amount_3h")
        is None
    )


def test_select_variable_source_wind_10m():
    cand_no_wind = CanonicalCandidate(
        cycle_time=_dt(2026, 9, 10, 0),
        lead_time_hours=6,
        run_id="run_no_wind",
        store_path="s3://store_no_wind",
        product_types=frozenset({"surface"}),
        variables=frozenset({"temperature_2m"}),
    )
    cand_partial_wind = CanonicalCandidate(
        cycle_time=_dt(2026, 9, 10, 0),
        lead_time_hours=12,
        run_id="run_partial_wind",
        store_path="s3://store_partial",
        product_types=frozenset({"surface"}),
        variables=frozenset({"temperature_2m", "wind_u_10m"}),
    )
    cand_full_wind = CanonicalCandidate(
        cycle_time=_dt(2026, 9, 9, 18),
        lead_time_hours=18,
        run_id="run_full_wind",
        store_path="s3://store_full",
        product_types=frozenset({"surface"}),
        variables=frozenset({"temperature_2m", "wind_u_10m", "wind_v_10m"}),
    )

    # Neither cand_no_wind nor cand_partial_wind matches
    assert (
        select_variable_source(
            [cand_no_wind, cand_partial_wind], "wind_10m"
        )
        is None
    )

    # cand_full_wind matches
    w_src = select_variable_source(
        [cand_no_wind, cand_partial_wind, cand_full_wind], "wind_10m"
    )
    assert w_src == cand_full_wind

    # Ordinary variable missing returns None
    assert (
        select_variable_source([cand_no_wind], "nonexistent_var")
        is None
    )


def test_select_canonical_sources_bulk():
    vt1 = _dt(2026, 9, 10, 6)
    vt2 = _dt(2026, 9, 10, 12)
    vt_empty = _dt(2026, 9, 10, 18)

    cand_vt1_0 = CanonicalCandidate(
        cycle_time=_dt(2026, 9, 10, 6),
        lead_time_hours=0,
        run_id="run_vt1_lead0",
        store_path="s3://store0",
        product_types=frozenset({"surface"}),
        variables=frozenset(
            {
                "temperature_2m",
                "precipitation_amount_3h",
                "cloud_cover_3h",
                "crain",
                "wind_u_10m",
                "wind_v_10m",
            }
        ),
    )
    cand_vt1_6 = CanonicalCandidate(
        cycle_time=_dt(2026, 9, 10, 0),
        lead_time_hours=6,
        run_id="run_vt1_lead6",
        store_path="s3://store1",
        product_types=frozenset({"surface"}),
        variables=frozenset(
            {
                "temperature_2m",
                "precipitation_amount_3h",
                "cloud_cover_3h",
                "crain",
                "wind_u_10m",
                "wind_v_10m",
            }
        ),
    )
    cand_vt2_6 = CanonicalCandidate(
        cycle_time=_dt(2026, 9, 10, 6),
        lead_time_hours=6,
        run_id="run_vt2_lead6",
        store_path="s3://store0",
        product_types=frozenset({"surface"}),
        variables=frozenset(
            {
                "temperature_2m",
                "precipitation_amount_3h",
                "cloud_cover_3h",
                "crain",
                "wind_u_10m",
                "wind_v_10m",
            }
        ),
    )

    cands_by_valid = {
        vt1: [cand_vt1_0, cand_vt1_6],
        vt2: [cand_vt2_6],
        vt_empty: [],
    }

    # Test filtering with start_valid_time and target_valid_times
    res = select_canonical_sources_bulk(
        cands_by_valid,
        variables=[
            "temperature_2m",
            "precipitation_amount_3h",
            "crain",
            "cloud_cover_3h",
            "wind_10m",
        ],
        target_valid_times=[vt1, vt2, vt_empty],
        start_valid_time=vt1,
    )

    assert res.anchors[vt1] == cand_vt1_0
    assert res.anchors[vt2] == cand_vt2_6
    assert res.variable_sources[("temperature_2m", vt1)] == cand_vt1_0
    assert res.variable_sources[("precipitation_amount_3h", vt1)] == cand_vt1_6
    assert res.variable_sources[("crain", vt1)] == cand_vt1_6
    assert res.variable_sources[("cloud_cover_3h", vt1)] == cand_vt1_6
    assert res.variable_sources[("wind_10m", vt1)] == cand_vt1_0

    # vt1 had anchor lead 0, so interval fallbacks and companions are tracked
    assert res.interval_fallbacks[("precipitation_amount_3h", vt1)] == cand_vt1_6
    assert res.interval_fallbacks[("cloud_cover_3h", vt1)] == cand_vt1_6
    assert res.companion_dependencies[("crain", vt1)] == cand_vt1_6

    # vt2 had anchor lead 6, so no interval fallbacks
    assert ("precipitation_amount_3h", vt2) not in res.interval_fallbacks

    # Test with target_valid_times filter excluding vt1
    res_filtered = select_canonical_sources_bulk(
        cands_by_valid,
        variables=["temperature_2m"],
        target_valid_times=[vt2],
    )
    assert vt1 not in res_filtered.anchors
    assert vt2 in res_filtered.anchors

    # Test with start_valid_time after vt1
    res_start = select_canonical_sources_bulk(
        cands_by_valid,
        variables=["temperature_2m"],
        start_valid_time=vt2,
    )
    assert vt1 not in res_start.anchors
    assert vt2 in res_start.anchors


def test_filter_candidates_by_physical_fence():
    vt = _dt(2026, 9, 10, 6)
    cand_det = CanonicalCandidate(
        cycle_time=_dt(2026, 9, 10, 0),
        lead_time_hours=6,
        run_id="run_det",
        store_path="s3://store_det",
        product_types=frozenset({"surface"}),
        variables=frozenset({"temperature_2m", "precipitation_rate"}),
    )
    cand_gefs = CanonicalCandidate(
        cycle_time=_dt(2026, 9, 10, 0),
        lead_time_hours=6,
        run_id="run_gefs",
        store_path="s3://store_gefs",
        product_types=frozenset({"surface", "ensemble_mean"}),
        variables=frozenset({"temperature_2m"}),
        member_indices=tuple(range(1, 31)),
    )

    # Empty fenced keys returns input directly
    cands_map = {vt: [cand_det, cand_gefs]}
    assert filter_candidates_by_physical_fence(cands_map, set()) == cands_map

    # Fence deterministic variable temperature_2m
    fenced_temp = {("run_det", 6, "temperature_2m", "det", 0)}
    filtered_temp = filter_candidates_by_physical_fence(
        {vt: [cand_det]}, fenced_temp, is_ensemble=False
    )
    assert vt in filtered_temp
    assert "temperature_2m" not in filtered_temp[vt][0].variables
    assert "precipitation_rate" in filtered_temp[vt][0].variables

    # Fence all variables of deterministic candidate -> candidate dropped
    fenced_all_det = {
        ("run_det", 6, "temperature_2m", "det", 0),
        ("run_det", 6, "precipitation_rate", "det", 0),
    }
    filtered_all_det = filter_candidates_by_physical_fence(
        {vt: [cand_det]}, fenced_all_det, is_ensemble=False
    )
    assert vt not in filtered_all_det

    # Fence 5 GEFS members (leaves 25 members < 26 threshold -> candidate dropped)
    fenced_gefs = {
        ("run_gefs", 6, "temperature_2m", "mem", m)
        for m in (1, 2, 3, 4, 5)
    }
    filtered_gefs = filter_candidates_by_physical_fence(
        {vt: [cand_gefs]}, fenced_gefs, is_ensemble=True, expected_members=30
    )
    assert vt not in filtered_gefs

    # Fence 4 GEFS members (leaves 26 members >= 26 threshold -> kept with 26 members)
    fenced_gefs_4 = {
        ("run_gefs", 6, "temperature_2m", "mem", m)
        for m in (1, 2, 3, 4)
    }
    filtered_gefs_4 = filter_candidates_by_physical_fence(
        {vt: [cand_gefs]}, fenced_gefs_4, is_ensemble=True, expected_members=30
    )
    assert vt in filtered_gefs_4
    assert len(filtered_gefs_4[vt][0].member_indices or ()) == 26

    # Fence mean product for GEFS candidate
    fenced_mean = {("run_gefs", 6, "temperature_2m", "mean", -1)}
    filtered_mean = filter_candidates_by_physical_fence(
        {vt: [cand_gefs]}, fenced_mean, is_ensemble=True, expected_members=30
    )
    assert "ensemble_mean" not in filtered_mean[vt][0].product_types
