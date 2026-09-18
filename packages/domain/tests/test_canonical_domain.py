"""Unit tests for pure canonical domain logic (packages/domain/src/domain/canonical.py).

Enforces 100% test coverage across all branches, helpers, and perspectives.
"""

from __future__ import annotations

import random
import time
from datetime import UTC, datetime, timedelta

from domain.canonical import (
    CanonicalCandidate,
    _ensure_utc,
    build_fence_index,
    filter_candidates_by_fence_index,
    filter_candidates_by_physical_fence,
    select_canonical_anchor,
    select_canonical_sources_bulk,
    select_variable_source,
)
from domain.coverage import is_lead_servable


def _dt(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, 0, tzinfo=UTC)


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


def _reference_filter_candidates_by_physical_fence(
    candidates_by_valid: dict[datetime, list[CanonicalCandidate]],
    fenced_keys: set[tuple[str, int, str, str, int]],
    is_ensemble: bool = False,
    expected_members: int = 30,
) -> dict[datetime, list[CanonicalCandidate]]:
    """Literal transliteration of the pre-optimisation linear-scan implementation.

    This is the semantic ground truth the indexed implementation is asserted
    against; it deliberately keeps the O(|fenced_keys|) ``any(...)`` scans.
    """
    if not fenced_keys:
        return candidates_by_valid

    out: dict[datetime, list[CanonicalCandidate]] = {}
    for v_time, cands in candidates_by_valid.items():
        filtered_cands: list[CanonicalCandidate] = []
        for cand in cands:
            r_id = cand.run_id
            lead = cand.lead_time_hours

            active_vars = {
                v
                for v in cand.variables
                if not any(
                    fk[0] == r_id
                    and fk[1] == lead
                    and fk[2] == v
                    and fk[3] in ("det", "mean")
                    for fk in fenced_keys
                )
            }

            if is_ensemble and cand.member_indices is not None:
                m_indices = cand.member_indices
                avail_members = [
                    m
                    for m in m_indices
                    if not any(
                        fk[0] == r_id
                        and fk[1] == lead
                        and fk[3] == "mem"
                        and fk[4] == m
                        for fk in fenced_keys
                    )
                ]
                if not is_lead_servable(len(avail_members), expected_members):
                    continue
                mean_fenced = any(
                    fk[0] == r_id and fk[1] == lead and fk[3] == "mean"
                    for fk in fenced_keys
                )
                prod_types = (
                    cand.product_types - {"ensemble_mean"}
                    if mean_fenced
                    else cand.product_types
                )
                cand = CanonicalCandidate(
                    cycle_time=cand.cycle_time,
                    lead_time_hours=cand.lead_time_hours,
                    run_id=cand.run_id,
                    store_path=cand.store_path,
                    product_types=prod_types,
                    variables=frozenset(active_vars),
                    member_indices=tuple(avail_members),
                    status=cand.status,
                )
            else:
                if cand.variables and not active_vars:
                    continue
                cand = CanonicalCandidate(
                    cycle_time=cand.cycle_time,
                    lead_time_hours=cand.lead_time_hours,
                    run_id=cand.run_id,
                    store_path=cand.store_path,
                    product_types=cand.product_types,
                    variables=frozenset(active_vars),
                    member_indices=cand.member_indices,
                    status=cand.status,
                )
            filtered_cands.append(cand)
        if filtered_cands:
            out[v_time] = filtered_cands
    return out


def test_filter_candidates_matches_reference_on_randomised_inputs():
    """The indexed implementation must be output-identical to the scan baseline.

    Randomised differential test over mixed determinstic/mean/member fences,
    both ensemble and non-ensemble evaluation, and run/lead combinations that
    are fenced, unfenced, and absent from the candidate set.
    """
    rng = random.Random(20260917)
    all_vars = ["temperature_2m", "precipitation_rate", "wind_u_10m"]
    kinds = ["det", "mean", "mem"]

    # Guarantee every kind appears so both outcomes of each projection's
    # filter predicate are exercised deterministically.
    base_keys = [
        ("run_a", 6, "temperature_2m", "det", 0),
        ("run_a", 6, "temperature_2m", "mean", -1),
        ("run_a", 6, "temperature_2m", "mem", 1),
    ]

    for _ in range(200):
        fenced = set(base_keys)
        for _ in range(rng.randrange(0, 25)):
            fenced.add(
                (
                    rng.choice(["run_a", "run_b", "run_absent"]),
                    rng.choice([0, 6, 12]),
                    rng.choice(all_vars),
                    rng.choice(kinds),
                    rng.choice([-1, 0, 1, 2, 30, 31]),
                )
            )

        cands_map: dict[datetime, list[CanonicalCandidate]] = {}
        for idx in range(rng.randrange(1, 4)):
            vt = _dt(2026, 9, 10, 0) + timedelta(hours=6 * idx + rng.choice([0, 3]))
            members = tuple(range(1, rng.choice([31, 30, 5]) + 1))
            cands_map.setdefault(vt, []).append(
                CanonicalCandidate(
                    cycle_time=_dt(2026, 9, 10, 0),
                    lead_time_hours=rng.choice([0, 6, 12]),
                    run_id=rng.choice(["run_a", "run_b"]),
                    store_path="s3://store",
                    product_types=frozenset(
                        rng.choice([{"surface"}, {"surface", "ensemble_mean"}])
                    ),
                    variables=frozenset(
                        rng.sample(all_vars, rng.randrange(1, len(all_vars) + 1))
                    ),
                    member_indices=members if rng.random() < 0.6 else None,
                )
            )

        is_ensemble = rng.random() < 0.5
        expected_members = rng.choice([30, 5, 31])

        assert filter_candidates_by_physical_fence(
            cands_map, fenced, is_ensemble=is_ensemble, expected_members=expected_members
        ) == _reference_filter_candidates_by_physical_fence(
            cands_map, fenced, is_ensemble=is_ensemble, expected_members=expected_members
        )


def test_filter_candidates_scales_to_production_sized_fenced_set():
    """Guard the O(candidates x variables x |fenced_keys|) regression.

    Sized like the production worker revalidation (50k fenced units, 800
    ensemble candidates with 30 members). The linear-scan implementation needs
    roughly a minute on this input; the indexed one is sub-second, so the
    ceiling below fails loudly on any reintroduced scan.
    """
    fenced = {
        (f"run_{i % 12}", (i % 82) * 3, "temperature_2m", "mem", 1 + (i % 30))
        for i in range(50_000)
    }
    fenced |= {(f"run_{i}", 0, "temperature_2m", "mean", -1) for i in range(12)}

    cands_map: dict[datetime, list[CanonicalCandidate]] = {}
    for lead in range(0, 240, 3):
        for run in range(12):
            cands_map.setdefault(_dt(2026, 9, 10, 0) + timedelta(hours=lead), []).append(
                CanonicalCandidate(
                    cycle_time=_dt(2026, 9, 10, 0),
                    lead_time_hours=lead,
                    run_id=f"run_{run}",
                    store_path="s3://store",
                    product_types=frozenset({"surface", "ensemble_mean"}),
                    variables=frozenset({"temperature_2m", "precipitation_rate"}),
                    member_indices=tuple(range(1, 31)),
                )
            )

    started = time.perf_counter()
    filtered = filter_candidates_by_physical_fence(
        cands_map, fenced, is_ensemble=True, expected_members=30
    )
    elapsed = time.perf_counter() - started

    assert filtered, "expected the fenced set to retain some candidates"
    assert elapsed < 3.0, f"fence filter regressed to {elapsed:.2f}s"


def test_build_fence_index_matches_the_set_built_projection():
    """The index built from a stream must equal the index built from a set.

    Both paths run the same single projection implementation, so this pins the
    input-path equivalence that lets the worker hand a database cursor to
    ``build_fence_index`` instead of materialising the whole row set.
    """
    rng = random.Random(20260918)
    values = ["run_a", "run_b", "run_absent"]
    vars_ = ["temperature_2m", "wind_u_10m", "relative_humidity_2m"]
    kinds = ["det", "mean", "mem", "unrecognised"]

    for _ in range(200):
        fenced = {
            (
                rng.choice(values),
                rng.choice([0, 6, 12]),
                rng.choice(vars_),
                rng.choice(kinds),
                rng.choice([-1, 0, 1, 2, 30, 31]),
            )
            for _ in range(rng.randrange(0, 30))
        }
        from_set = build_fence_index(fenced)
        from_stream = build_fence_index(iter(sorted(fenced)))
        assert from_stream == from_set
        assert from_stream.has_keys == bool(fenced)


def test_build_fence_index_projects_each_target_kind():
    fenced = [
        ("run_a", 6, "temperature_2m", "det", 0),
        ("run_a", 6, "temperature_2m", "mean", -1),
        ("run_a", 6, "wind_u_10m", "mem", 7),
        ("run_a", 6, "wind_u_10m", "unrecognised", 0),
    ]
    index = build_fence_index(iter(fenced))

    assert index.variable_keys == {("run_a", 6, "temperature_2m")}
    assert index.member_keys == {("run_a", 6, 7)}
    assert index.mean_run_leads == {("run_a", 6)}
    assert index.has_keys is True


def test_build_fence_index_consumes_the_iterable_exactly_once_and_fully():
    consumed: list[int] = []

    def stream():
        for i in range(1000):
            consumed.append(i)
            yield ("run_a", i % 12, "temperature_2m", "mem", 1 + (i % 30))

    index = build_fence_index(stream())

    assert len(consumed) == 1000
    assert index.has_keys is True


def test_build_fence_index_deduplicates_string_components():
    """The projections must retain one object per distinct string value.

    A database driver hands back a fresh ``str`` per row, so the projections
    would otherwise hold one owned copy of ``run_id`` / ``variable_code`` per
    distinct key; at production scale that is the difference between roughly
    40 MB and 85 MB retained for the same fence semantics.
    """
    run_a = "".join(["run_", "dedup"])
    run_b = "".join(["run_", "dedup"])
    var_a = "".join(["temperature", "_2m"])
    var_b = "".join(["temperature", "_2m"])
    assert run_a is not run_b and run_a == run_b
    assert var_a is not var_b and var_a == var_b

    index = build_fence_index(
        [
            (run_a, 6, var_a, "det", 0),
            (run_b, 6, var_b, "det", 0),
            (run_b, 6, "wind_u_10m", "mem", 1),
            (run_b, 6, var_b, "mean", -1),
        ]
    )

    for r_id, _lead, var in index.variable_keys:
        assert r_id is run_a
        if var == var_a:
            assert var is var_a
    for r_id, _lead, _member in index.member_keys:
        assert r_id is run_a
    for r_id, _lead in index.mean_run_leads:
        assert r_id is run_a


def test_filter_candidates_by_fence_index_parity_with_the_raw_set_entry_point():
    """The streamed entry point must match the raw-set entry point, which must
    itself still match the linear-scan reference implementation.

    Chain-asserted over the same randomised input family the projection
    equivalence test uses, so the three implementations cannot drift apart.
    """
    rng = random.Random(20260919)
    all_vars = ["temperature_2m", "precipitation_rate", "wind_u_10m"]
    kinds = ["det", "mean", "mem"]

    for _ in range(200):
        fenced = {
            (rng.choice(["run_a", "run_b", "run_absent"]), rng.choice([0, 6, 12]),
             rng.choice(all_vars), rng.choice(kinds), rng.choice([-1, 0, 1, 30, 31]))
            for _ in range(rng.randrange(0, 25))
        }

        cands_map: dict[datetime, list[CanonicalCandidate]] = {}
        for idx in range(rng.randrange(1, 4)):
            vt = _dt(2026, 9, 10, 0) + timedelta(hours=6 * idx)
            cands_map.setdefault(vt, []).append(
                CanonicalCandidate(
                    cycle_time=_dt(2026, 9, 10, 0),
                    lead_time_hours=rng.choice([0, 6, 12]),
                    run_id=rng.choice(["run_a", "run_b"]),
                    store_path="s3://store",
                    product_types=frozenset(
                        rng.choice([{"surface"}, {"surface", "ensemble_mean"}])
                    ),
                    variables=frozenset(
                        rng.sample(all_vars, rng.randrange(1, len(all_vars) + 1))
                    ),
                    member_indices=tuple(range(1, 31)) if rng.random() < 0.6 else None,
                )
            )

        is_ensemble = rng.random() < 0.5
        expected_members = rng.choice([30, 31])

        from_raw_set = filter_candidates_by_physical_fence(
            cands_map, fenced, is_ensemble=is_ensemble, expected_members=expected_members
        )
        from_streamed_index = filter_candidates_by_fence_index(
            cands_map,
            build_fence_index(iter(sorted(fenced))),
            is_ensemble=is_ensemble,
            expected_members=expected_members,
        )
        from_reference = _reference_filter_candidates_by_physical_fence(
            cands_map, fenced, is_ensemble=is_ensemble, expected_members=expected_members
        )

        assert from_streamed_index == from_raw_set
        assert from_streamed_index == from_reference


def test_filter_candidates_by_fence_index_empty_fence_semantics():
    """Empty-input behaviour, for both entry points.

    The early return keys on the *input* being empty, not on the projections
    being empty: an input whose every ``target_kind`` is unrecognised projects
    to three empty sets and must still take the rebuild path.
    """
    vt = _dt(2026, 9, 10, 6)
    cand = CanonicalCandidate(
        cycle_time=_dt(2026, 9, 10, 0),
        lead_time_hours=6,
        run_id="run_det",
        store_path="s3://store_det",
        product_types=frozenset({"surface"}),
        variables=frozenset({"temperature_2m"}),
    )
    cands_map = {vt: [cand]}

    empty_index = build_fence_index(iter(()))
    assert empty_index.has_keys is False
    assert filter_candidates_by_fence_index(cands_map, empty_index) is cands_map
    assert filter_candidates_by_physical_fence(cands_map, set()) is cands_map

    unknown_index = build_fence_index([("run_det", 6, "temperature_2m", "unrecognised", 0)])
    assert unknown_index.has_keys is True
    assert not unknown_index.variable_keys
    assert not unknown_index.member_keys
    assert not unknown_index.mean_run_leads

    rebuilt = filter_candidates_by_fence_index(cands_map, unknown_index)
    assert rebuilt == cands_map
    assert rebuilt is not cands_map

