from __future__ import annotations

import pandas as pd

from model_health import assess_delivery_health
from prediction_validation import _direction, rolling_7day_scorecard


def test_hold_tolerance_is_a_fraction_not_a_percent() -> None:
    # 0.001 == 0.1%, which is inside the configured 0.005 (0.5%) flat band.
    assert _direction(0.001, 0.005) == "FLAT"
    assert _direction(-0.001, 0.005) == "FLAT"
    assert _direction(0.010, 0.005) == "UP"
    assert _direction(-0.010, 0.005) == "DOWN"


def test_legacy_percent_tolerance_is_still_accepted() -> None:
    assert _direction(0.001, 0.5) == "FLAT"
    assert _direction(0.010, 0.5) == "UP"


def _rows(count: int) -> pd.DataFrame:
    start = pd.Timestamp("2026-09-01 15:30:00")
    return pd.DataFrame([
        {
            "validation_status": "VALIDATED",
            "decision_timestamp": (start + pd.Timedelta(days=i)).isoformat(),
            "direction_match_d1": True,
            "percentage_error_d1": 0.01,
            "probability_up_next_session": 0.7,
            "actual_direction_d1": "UP",
            "signal": "BUY",
        }
        for i in range(count)
    ])


def test_hold_rows_are_not_scored_as_directional_predictions() -> None:
    rows = _rows(10)
    rows.loc[0:8, "signal"] = "HOLD"
    rows.loc[0:8, "actual_direction_d1"] = "FLAT"
    score = rolling_7day_scorecard(rows.tail(7))
    # The seven-row window holds six HOLD rows and one BUY row. HOLD has no
    # directional claim, so the gate cannot pass on seven completed rows.
    assert score["scored_predictions"] == 1
    assert score["directional_accuracy_scored_samples"] == 1
    assert score["status"] == "INSUFFICIENT_HISTORY"


def test_delivery_health_excludes_hold_from_accuracy() -> None:
    rows = _rows(10)
    rows.loc[0:5, "signal"] = "HOLD"
    rows.loc[0:5, "actual_direction_d1"] = "UP"
    health = assess_delivery_health(rows)
    assert health["sample_count"] == 4
    assert health["hold_samples"] == 3
    assert health["retraining_min_samples"] >= 10
    assert health["retraining_candidate"] is False
