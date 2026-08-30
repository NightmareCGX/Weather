"""Unit tests for the NOMADS .idx parser and product-aware record selector."""

from __future__ import annotations

import pytest

from ingestion.providers.noaa.idx_parser import (
    IdxParseError,
    SelectionStatus,
    parse_idx,
    select_gefs_records,
    select_gfs_records,
    select_records,
)

SAMPLE_GFS_IDX = """
1:0:d=2026082600:PRMSL:mean sea level:6 hour fcst:
2:996400:d=2026082600:CLWMR:1 hybrid level:6 hour fcst:
580:420107954:d=2026082600:ICETK:surface:6 hour fcst:
581:420191661:d=2026082600:TMP:2 m above ground:6 hour fcst:
582:420700250:d=2026082600:SPFH:2 m above ground:6 hour fcst:
593:427947306:d=2026082600:PRATE:surface:6 hour fcst:
594:428544080:d=2026082600:CPRAT:surface:0-6 hour ave fcst:
595:429291423:d=2026082600:PRATE:surface:0-6 hour ave fcst:
596:430008042:d=2026082600:APCP:surface:0-6 hour acc fcst:
"""

SAMPLE_GEFS_IDX = """
1:0:d=2026082600:VIS:surface:6 hour fcst:ENS=+1
2:330075:d=2026082600:GUST:surface:6 hour fcst:ENS=+1
3:873710:d=2026082600:MSLET:mean sea level:6 hour fcst:ENS=+1
9:3695387:d=2026082600:ICETK:surface:6 hour fcst:ENS=+1
10:3769502:d=2026082600:TMP:2 m above ground:6 hour fcst:ENS=+1
11:4200404:d=2026082600:DPT:2 m above ground:6 hour fcst:ENS=+1
18:8446969:d=2026082600:APCP:surface:0-6 hour acc fcst:ENS=+1
38:18051651:d=2026082600:PRMSL:mean sea level:6 hour fcst:ENS=+1
"""


def _make_continuous_idx(lines: list[str]) -> str:
    """Helper to renumber lines so record_number is 1..N for strict parser."""
    out = []
    for i, line in enumerate(lines):
        parts = line.split(":")
        parts[0] = str(i + 1)
        out.append(":".join(parts))
    return "\n".join(out)


def test_parse_idx_valid_gfs_lines() -> None:
    raw_lines = [
        "1:0:d=2026082600:PRMSL:mean sea level:6 hour fcst:",
        "2:996400:d=2026082600:CLWMR:1 hybrid level:6 hour fcst:",
        "3:420191661:d=2026082600:TMP:2 m above ground:6 hour fcst:",
        "4:420700250:d=2026082600:PRATE:surface:6 hour fcst:",
        "5:428544080:d=2026082600:PRATE:surface:0-6 hour ave fcst:",
    ]
    text = "\n".join(raw_lines)
    records = parse_idx(text)
    assert len(records) == 5

    # Check first record
    assert records[0].record_number == 1
    assert records[0].start_offset == 0
    assert records[0].end_offset == 996399
    assert records[0].byte_length == 996400
    assert records[0].parameter == "PRMSL"
    assert records[0].level_description == "mean sea level"

    # Check 3rd record (TMP)
    assert records[2].record_number == 3
    assert records[2].start_offset == 420191661
    assert records[2].end_offset == 420700249
    assert records[2].byte_length == 508589
    assert records[2].parameter == "TMP"
    assert records[2].level_description == "2 m above ground"

    # Check final record (unbounded end_offset)
    assert records[4].record_number == 5
    assert records[4].start_offset == 428544080
    assert records[4].end_offset is None
    assert records[4].byte_length is None


def test_parse_idx_valid_gefs() -> None:
    raw_lines = [
        "1:0:d=2026082600:VIS:surface:6 hour fcst:ENS=+1",
        "2:330075:d=2026082600:TMP:2 m above ground:6 hour fcst:ENS=+1",
        "3:873710:d=2026082600:APCP:surface:0-6 hour acc fcst:ENS=+1",
    ]
    text = "\n".join(raw_lines)
    records = parse_idx(text)
    assert len(records) == 3
    assert records[1].parameter == "TMP"
    assert records[1].ensemble_description == "ENS=+1"
    assert records[1].start_offset == 330075
    assert records[1].end_offset == 873709


def test_parse_idx_rejects_empty_or_whitespace() -> None:
    with pytest.raises(IdxParseError, match="Empty"):
        parse_idx("")
    with pytest.raises(IdxParseError, match="Empty"):
        parse_idx("   \n\n\t  ")


def test_parse_idx_rejects_malformed_fields() -> None:
    # Less than 6 fields
    with pytest.raises(IdxParseError, match="expected at least 6 fields"):
        parse_idx("1:0:d=2026082600:TMP:surface")

    # Non-integer record number
    with pytest.raises(IdxParseError, match="Invalid record number"):
        parse_idx("X:0:d=2026082600:TMP:surface:6 hour fcst:")

    # Non-integer offset
    with pytest.raises(IdxParseError, match="Invalid byte offset"):
        parse_idx("1:OFFSET:d=2026082600:TMP:surface:6 hour fcst:")


def test_parse_idx_rejects_negative_or_nonzero_first_offset() -> None:
    with pytest.raises(IdxParseError, match="must start at byte offset 0"):
        parse_idx("1:100:d=2026082600:TMP:surface:6 hour fcst:")

    with pytest.raises(IdxParseError, match="Negative byte offset"):
        parse_idx("1:-10:d=2026082600:TMP:surface:6 hour fcst:")


def test_parse_idx_rejects_non_monotonic_or_duplicate_offsets() -> None:
    # Duplicate offsets
    raw = "1:0:d=2026082600:TMP:surface:6 hour fcst:\n2:0:d=2026082600:TMP:surface:6 hour fcst:"
    with pytest.raises(IdxParseError, match="Non-increasing byte offset"):
        parse_idx(raw)

    # Decreasing offset
    raw2 = "1:0:d=2026082600:TMP:surface:6 hour fcst:\n2:500:d=2026082600:TMP:surface:6 hour fcst:\n3:400:d=2026082600:TMP:surface:6 hour fcst:"
    with pytest.raises(IdxParseError, match="Non-increasing byte offset"):
        parse_idx(raw2)


def test_parse_idx_rejects_non_sequential_record_numbers() -> None:
    raw = "1:0:d=2026082600:TMP:surface:6 hour fcst:\n3:500:d=2026082600:TMP:surface:6 hour fcst:"
    with pytest.raises(IdxParseError, match="Non-sequential record number"):
        parse_idx(raw)


def test_select_gfs_records_success() -> None:
    lines = [
        "1:0:d=2026082600:PRMSL:mean sea level:6 hour fcst:",
        "2:500:d=2026082600:TMP:2 m above ground:6 hour fcst:",
        "3:1500:d=2026082600:SPFH:2 m above ground:6 hour fcst:",
        "4:2500:d=2026082600:PRATE:surface:6 hour fcst:",
        "5:3500:d=2026082600:PRATE:surface:0-6 hour ave fcst:",
        "6:4500:d=2026082600:RH:2 m above ground:6 hour fcst:",
        "7:5500:d=2026082600:GUST:surface:6 hour fcst:",
        "8:6500:d=2026082600:VIS:surface:6 hour fcst:",
        "9:7500:d=2026082600:SNOD:surface:6 hour fcst:",
        "10:8500:d=2026082600:UGRD:10 m above ground:6 hour fcst:",
        "11:9500:d=2026082600:VGRD:10 m above ground:6 hour fcst:",
        "12:10500:d=2026082600:APCP:surface:0-6 hour acc fcst:",
    ]
    records = parse_idx("\n".join(lines))
    result = select_gfs_records(records, lead_time_hours=6)

    assert result.is_valid
    assert len(result.selected_records) == 8
    assert result.missing_required == ()
    assert result.ambiguous == ()

    # Selected records are strictly sorted by start_offset
    params = [r.parameter for r in result.selected_records]
    assert params == ["TMP", "PRATE", "RH", "GUST", "VIS", "SNOD", "UGRD", "VGRD"]
    assert result.selected_records[0].start_offset == 500
    assert result.selected_records[0].end_offset == 1499
    assert result.selected_records[1].start_offset == 2500
    assert result.selected_records[1].end_offset == 3499


def test_select_gfs_records_lead_zero_analysis() -> None:
    lines = [
        "1:0:d=2026082600:PRMSL:mean sea level:anl:",
        "2:500:d=2026082600:TMP:2 m above ground:anl:",
        "3:1500:d=2026082600:PRATE:surface:anl:",
        "4:2500:d=2026082600:RH:2 m above ground:anl:",
        "5:3500:d=2026082600:GUST:surface:anl:",
        "6:4500:d=2026082600:VIS:surface:anl:",
        "7:5500:d=2026082600:SNOD:surface:anl:",
        "8:6500:d=2026082600:UGRD:10 m above ground:anl:",
        "9:7500:d=2026082600:VGRD:10 m above ground:anl:",
    ]
    records = parse_idx("\n".join(lines))
    result = select_gfs_records(records, lead_time_hours=0)
    assert result.is_valid
    assert len(result.selected_records) == 8
    params = [r.parameter for r in result.selected_records]
    assert params == ["TMP", "PRATE", "RH", "GUST", "VIS", "SNOD", "UGRD", "VGRD"]


def test_select_gfs_records_excludes_averaged_and_accumulated_prate() -> None:
    lines = [
        "1:0:d=2026082600:TMP:2 m above ground:6 hour fcst:",
        "2:500:d=2026082600:PRATE:surface:0-6 hour ave fcst:",
        "3:1500:d=2026082600:PRATE:surface:6 hour acc fcst:",
        "4:2500:d=2026082600:PRATE:surface:6 hour fcst:",
    ]
    records = parse_idx("\n".join(lines))
    result = select_gfs_records(
        records,
        lead_time_hours=6,
        variables=("temperature_2m", "precipitation_rate"),
    )
    assert result.is_valid
    assert len(result.selected_records) == 2
    # Only instant PRATE is selected
    prate_rec = result.variable_selections["precipitation_rate"].record
    assert prate_rec is not None
    assert prate_rec.forecast_description == "6 hour fcst"
    assert prate_rec.start_offset == 2500


def test_select_gfs_missing_required_variable() -> None:
    lines = [
        "1:0:d=2026082600:TMP:2 m above ground:6 hour fcst:",
        "2:500:d=2026082600:SPFH:2 m above ground:6 hour fcst:",
    ]
    records = parse_idx("\n".join(lines))
    result = select_gfs_records(
        records,
        lead_time_hours=6,
        variables=("temperature_2m", "precipitation_rate"),
    )
    assert not result.is_valid
    assert "precipitation_rate" in result.missing_required
    assert (
        result.variable_selections["precipitation_rate"].status
        == SelectionStatus.REQUIRED_BUT_MISSING
    )


def test_select_gfs_ambiguous_variable() -> None:
    lines = [
        "1:0:d=2026082600:TMP:2 m above ground:6 hour fcst:",
        "2:500:d=2026082600:TMP:2 m above ground:6 hour fcst:",  # Duplicate matching record
        "3:1500:d=2026082600:PRATE:surface:6 hour fcst:",
    ]
    records = parse_idx("\n".join(lines))
    result = select_gfs_records(
        records,
        lead_time_hours=6,
        variables=("temperature_2m", "precipitation_rate"),
    )
    assert not result.is_valid
    assert "temperature_2m" in result.ambiguous
    assert (
        result.variable_selections["temperature_2m"].status
        == SelectionStatus.AMBIGUOUS
    )


def test_select_gefs_records_success() -> None:
    lines = [
        "1:0:d=2026082600:VIS:surface:6 hour fcst:ENS=+17",
        "2:500:d=2026082600:TMP:2 m above ground:6 hour fcst:ENS=+17",
        "3:1500:d=2026082600:GUST:surface:6 hour fcst:ENS=+17",
        "4:2500:d=2026082600:RH:2 m above ground:6 hour fcst:ENS=+17",
        "5:3500:d=2026082600:SNOD:surface:6 hour fcst:ENS=+17",
        "6:4500:d=2026082600:UGRD:10 m above ground:6 hour fcst:ENS=+17",
        "7:5500:d=2026082600:VGRD:10 m above ground:6 hour fcst:ENS=+17",
        "8:6500:d=2026082600:APCP:surface:0-6 hour acc fcst:ENS=+17",
    ]
    records = parse_idx("\n".join(lines))
    result = select_gefs_records(records, member=17, lead_time_hours=6)
    assert result.is_valid
    assert len(result.selected_records) == 7
    params = [r.parameter for r in result.selected_records]
    assert params == ["VIS", "TMP", "GUST", "RH", "SNOD", "UGRD", "VGRD"]
    for r in result.selected_records:
        assert r.ensemble_description == "ENS=+17"
    # PRATE is classified as UNSUPPORTED, not missing_required
    assert (
        result.variable_selections["precipitation_rate"].status
        == SelectionStatus.UNSUPPORTED
    )
    assert "precipitation_rate" in result.unsupported
    assert result.missing_required == ()


def test_select_gefs_member_mismatch() -> None:
    lines = [
        "1:0:d=2026082600:VIS:surface:6 hour fcst:ENS=+01",
        "2:500:d=2026082600:TMP:2 m above ground:6 hour fcst:ENS=+01",
    ]
    records = parse_idx("\n".join(lines))
    # Request member 17 when index contains member 1
    result = select_gefs_records(records, member=17, lead_time_hours=6)
    assert not result.is_valid
    assert "temperature_2m" in result.missing_required


def test_select_records_dispatch() -> None:
    gfs_lines = [
        "1:0:d=2026082600:TMP:2 m above ground:6 hour fcst:",
        "2:500:d=2026082600:PRATE:surface:6 hour fcst:",
        "3:1500:d=2026082600:RH:2 m above ground:6 hour fcst:",
        "4:2500:d=2026082600:GUST:surface:6 hour fcst:",
        "5:3500:d=2026082600:VIS:surface:6 hour fcst:",
        "6:4500:d=2026082600:SNOD:surface:6 hour fcst:",
        "7:5500:d=2026082600:UGRD:10 m above ground:6 hour fcst:",
        "8:6500:d=2026082600:VGRD:10 m above ground:6 hour fcst:",
    ]
    records = parse_idx("\n".join(gfs_lines))
    res_gfs = select_records("gfs", records, lead_time_hours=6)
    assert res_gfs.is_valid
    assert len(res_gfs.selected_records) == 8

    # Test explicit variable subset
    res_subset = select_records(
        "gfs",
        records,
        lead_time_hours=6,
        variables=("temperature_2m", "wind_gust"),
    )
    assert res_subset.is_valid
    assert len(res_subset.selected_records) == 2

    with pytest.raises(ValueError, match="member must be provided"):
        select_records("gefs", records, lead_time_hours=6)

    with pytest.raises(ValueError, match="Unsupported model"):
        select_records("ecmwf", records, lead_time_hours=6)
