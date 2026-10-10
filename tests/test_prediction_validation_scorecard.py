from __future__ import annotations

import pandas as pd

from prediction_validation import rolling_7day_scorecard, validation_status_for_sample_count


def _validation_rows(count: int, missing_direction_match_at: int | None = None) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-09-01 15:30:00")
    for i in range(count):
        rows.append({
            "validation_status": "VALIDATED",
            "decision_timestamp": (start + pd.Timedelta(days=i)).isoformat(),
            "direction_match_d1": None if i == missing_direction_match_at else True,
            "percentage_error_d1": 0.01,
            "probability_up_next_session": 0.7,
            "actual_direction_d1": "UP",
            "signal": "BUY",
        })
    return pd.DataFrame(rows)


def test_three_scored_predictions_are_not_reported_as_pass() -> None:
    score = rolling_7day_scorecard(_validation_rows(3))
    assert score["status"] == "INSUFFICIENT_HISTORY"
    assert score["history_sufficient"] is False
    assert score["completed_predictions"] == 3
    assert score["scored_predictions"] == 3
    assert score["required_scored_predictions"] == 7


def test_seven_scored_predictions_can_pass_the_history_gate() -> None:
    score = rolling_7day_scorecard(_validation_rows(7))
    assert score["status"] == "PASS"
    assert score["history_sufficient"] is True
    assert score["scored_predictions"] == 7


def test_seven_rows_with_one_unscored_outcome_remain_insufficient() -> None:
    score = rolling_7day_scorecard(_validation_rows(7, missing_direction_match_at=0))
    assert score["status"] == "INSUFFICIENT_HISTORY"
    assert score["completed_predictions"] == 7
    assert score["scored_predictions"] == 6

def test_shared_history_status_gate_requires_minimum_sample_count() -> None:
    assert validation_status_for_sample_count(0, 7) == "NO_DATA"
    assert validation_status_for_sample_count(3, 7) == "INSUFFICIENT_HISTORY"
    assert validation_status_for_sample_count(7, 7) == "PASS"
