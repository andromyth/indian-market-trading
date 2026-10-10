from __future__ import annotations

import pandas as pd

from delivery_policy_update import (
    MIN_HOLDOUT_SESSIONS,
    MIN_UNIQUE_SESSIONS,
    _date_block_split,
    _gate_pass,
)


def test_date_block_split_keeps_each_session_entirely_in_one_partition():
    days = pd.bdate_range("2026-01-05", periods=25)
    rows = []
    for day_no, day in enumerate(days):
        for symbol_no in range(6):
            rows.append({
                "symbol": f"SYM{symbol_no}",
                "decision_timestamp": pd.Timestamp(day.date()).tz_localize("Asia/Kolkata").replace(hour=15, minute=30).isoformat(),
                "probability_up_next_session": 0.9 if symbol_no % 2 == 0 else 0.1,
                "predicted_return_next_session": 0.01 if symbol_no % 2 == 0 else -0.01,
                "actual_direction_d1": "UP" if symbol_no % 2 == 0 else "DOWN",
                "row_number": day_no * 6 + symbol_no,
            })
    data = pd.DataFrame(rows)
    split, total_sessions, holdout_sessions = _date_block_split(data)
    assert split is not None
    tune, hold = split
    assert total_sessions >= MIN_UNIQUE_SESSIONS
    assert holdout_sessions >= MIN_HOLDOUT_SESSIONS
    tune_dates = set(pd.to_datetime(tune["decision_timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d"))
    hold_dates = set(pd.to_datetime(hold["decision_timestamp"], utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d"))
    assert tune_dates.isdisjoint(hold_dates)
    assert len(tune) + len(hold) == len(data)
    assert max(tune_dates) < min(hold_dates)


def test_many_rows_from_one_session_do_not_count_as_enough_validation_history():
    stamp = pd.Timestamp("2026-01-05 15:30", tz="Asia/Kolkata").isoformat()
    data = pd.DataFrame({
        "decision_timestamp": [stamp] * 100,
        "probability_up_next_session": [0.9] * 100,
        "predicted_return_next_session": [0.01] * 100,
        "actual_direction_d1": ["UP"] * 100,
    })
    split, total_sessions, holdout_sessions = _date_block_split(data)
    assert split is None
    assert total_sessions == 1
    assert holdout_sessions == 0


def test_string_false_quality_gate_is_not_accepted():
    assert _gate_pass("False") is False
    assert _gate_pass("0") is False
    assert _gate_pass("true") is True
    assert _gate_pass(True) is True
    assert _gate_pass(None) is False
