from __future__ import annotations

import pytest

from selfdriving.stepwise_forecast import stepwise_horizon_path


def test_stepwise_references_are_previous_horizon_endpoints():
    predictions = [
        {"horizon_min": 15, "reference_close": 100, "expected_return": 0.02, "predicted_price": 102, "signal": "BUY", "probability_up": 0.7},
        {"horizon_min": 5, "reference_close": 100, "expected_return": 0.01, "predicted_price": 101, "signal": "BUY", "probability_up": 0.65},
        {"horizon_min": 10, "reference_close": 100, "expected_return": -0.01, "predicted_price": 99, "signal": "SELL", "probability_up": 0.4},
    ]

    rows = stepwise_horizon_path(predictions, current_reference_price=100)

    assert [row["horizon_min"] for row in rows] == [5, 10, 15]
    assert [row["reference_close"] for row in rows] == [100, 101, 99]
    assert [row["expected_move"] for row in rows] == [1, -2, 3]
    assert rows[0]["step_direction"] == "UP"
    assert rows[1]["step_direction"] == "DOWN"
    assert rows[2]["step_direction"] == "UP"
    assert rows[1]["expected_return"] == pytest.approx(99 / 101 - 1)
    assert rows[1]["reference_price_source"] == "PREVIOUS_PLUS_5_MIN_FORECAST_ENDPOINT"
    assert rows[2]["reference_price_source"] == "PREVIOUS_PLUS_10_MIN_FORECAST_ENDPOINT"


def test_stepwise_view_preserves_original_model_origin_fields_and_signals():
    original = [
        {"horizon_min": 5, "reference_close": 50, "expected_return": 0.1, "predicted_price": 55, "signal": "BUY", "probability_up": 0.8},
        {"horizon_min": 10, "reference_close": 50, "expected_return": 0.2, "predicted_price": 60, "signal": "HOLD", "probability_up": 0.55},
    ]

    rows = stepwise_horizon_path(original, current_reference_price=52)

    assert rows[0]["reference_close"] == 52
    assert rows[1]["reference_close"] == 55
    assert rows[1]["model_reference_close"] == 50
    assert rows[1]["model_expected_return_from_now"] == 0.2
    assert rows[1]["expected_return"] == pytest.approx(60 / 55 - 1)
    assert rows[1]["signal"] == "HOLD"
    assert rows[1]["probability_up"] == 0.55
    assert original[1]["reference_close"] == 50  # input is not mutated
    assert original[1]["expected_return"] == 0.2


def test_missing_previous_endpoint_blocks_only_that_step_reference():
    rows = stepwise_horizon_path(
        [
            {"horizon_min": 5, "reference_close": 100, "predicted_price": 101},
            {"horizon_min": 10, "reference_close": 100, "predicted_price": None},
            {"horizon_min": 15, "reference_close": 100, "predicted_price": 104},
            {"horizon_min": 20, "reference_close": 100, "predicted_price": 103},
        ],
        current_reference_price=100,
    )

    assert rows[1]["reference_close"] == 101
    assert rows[1]["expected_move"] is None
    assert rows[2]["reference_close"] is None
    assert rows[2]["step_direction"] == "UNAVAILABLE"
    assert rows[3]["reference_close"] == 104
    assert rows[3]["expected_move"] == -1


def test_first_row_falls_back_to_model_reference_when_live_price_absent():
    rows = stepwise_horizon_path([
        {"horizon_min": 5, "reference_close": 100, "predicted_price": 101}
    ])
    assert rows[0]["reference_close"] == 100
    assert rows[0]["reference_price_source"] == "MODEL_ORIGIN_REFERENCE_FALLBACK"
