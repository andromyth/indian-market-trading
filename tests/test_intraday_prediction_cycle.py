import pytest

from intraday_prediction_cycle import (
    select_prediction_complete_cycles,
    slots_between,
)


def _candidate(key: str, *, prediction: bool = True, status: str = "STORED",
               snapshot: bool = True) -> dict:
    return {
        "minute_key": key,
        "snapshot_stored": snapshot,
        "prediction_file_stored": prediction,
        "prediction_manifest_status": status,
    }


def test_selects_newest_complete_cycle_after_skipping_incomplete_ones():
    candidates = [
        _candidate("20261126_1235", prediction=False),
        _candidate("20261126_1230", status="PERSISTENCE_ERROR"),
        _candidate("20261126_1225"),
        _candidate("20261126_1220"),
    ]

    complete, skipped = select_prediction_complete_cycles(candidates, max_lookback_slots=6)

    assert [c["minute_key"] for c in complete] == [
        "20261126_1225", "20261126_1220",
    ]
    assert skipped == ["20261126_1235", "20261126_1230"]


def test_status_is_case_and_whitespace_insensitive():
    complete, skipped = select_prediction_complete_cycles(
        [_candidate("20261126_1200", status=" stored ")],
        max_lookback_slots=6,
    )
    assert [c["minute_key"] for c in complete] == ["20261126_1200"]
    assert skipped == []


def test_returns_no_cycles_when_all_predictions_are_missing():
    candidates = [
        _candidate("20261126_1235", prediction=False),
        _candidate("20261126_1230", prediction=False),
    ]

    complete, skipped = select_prediction_complete_cycles(candidates, max_lookback_slots=6)

    assert complete == []
    assert skipped == ["20261126_1235", "20261126_1230"]


def test_lookback_cap_bounds_the_search():
    candidates = [
        _candidate(f"20261126_{900 + offset:04d}", prediction=False)
        for offset in range(10)
    ]

    complete, skipped = select_prediction_complete_cycles(candidates, max_lookback_slots=3)

    assert complete == []
    assert len(skipped) == 3
    assert skipped == ["20261126_0900", "20261126_0901", "20261126_0902"]


def test_snapshots_without_stored_quote_do_not_consume_lookback():
    candidates = [
        _candidate("20261126_1235", snapshot=False),
        _candidate("20261126_1230"),
    ]

    complete, skipped = select_prediction_complete_cycles(candidates, max_lookback_slots=1)

    assert [c["minute_key"] for c in complete] == ["20261126_1230"]
    assert skipped == []


@pytest.mark.parametrize(
    "newest, chosen, expected",
    [
        ("20261126_1235", "20261126_1225", 2),
        ("20261126_1235", "20261126_1235", 0),
        ("20261126_0000", "20261126_2355", 0),
        ("not-a-minute", "20261126_1230", 0),
        (None, "20261126_1230", 0),
    ],
)
def test_slots_between(newest, chosen, expected):
    assert slots_between(newest, chosen) == expected
