"""Cross-language fixture consistency for the serving-window primitive (I20).

The JSON case table at ``tests/fixtures/serving_window_cases.json`` is the
single shared fixture consumed by BOTH this suite and the frontend Jest suite
(``services/frontend/src/lib/forecast/__tests__/serving-window-fixture.test.ts``),
locking the Python primitive in :mod:`domain.temporal` and the TS fallback
implementation (``computeServingStartValidTime`` / ``isServableValidTime``)
against semantic drift (Lifecycle V3 §11-11 interim lock).
"""

import json
from datetime import datetime
from pathlib import Path

import pytest
from domain.temporal import is_valid_time_protected, serving_start_valid_time

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "serving_window_cases.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.mark.parametrize(
    "case",
    FIXTURE["serving_start_cases"],
    ids=lambda c: c["now"],
)
def test_serving_start_matches_fixture(case: dict) -> None:
    now = _parse(case["now"])
    expected = _parse(case["expected_serving_start"])
    assert serving_start_valid_time(now) == expected


@pytest.mark.parametrize(
    "case",
    FIXTURE["protection_cases"],
    ids=lambda c: f"{c['valid_time']}@{c['now']}",
)
def test_protection_matches_fixture(case: dict) -> None:
    now = _parse(case["now"])
    valid_time = _parse(case["valid_time"])
    assert is_valid_time_protected(valid_time, now) is case["expected_protected"]
