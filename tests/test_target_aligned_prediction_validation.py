from __future__ import annotations

import pandas as pd

from prediction_validation import validate_delivery_prediction
import validate_intraday_30m_predictions as intraday


def _delivery_record(**updates):
    row = {
        "prediction_id": "delivery-1",
        "symbol": "SBIN",
        "mode": "Delivery",
        "decision_timestamp": "2026-10-19T15:30:00+05:30",
        "prediction_for_date": "2026-10-20",
        "signal": "BUY",
        "probability_up_next_session": 0.7,
        "predicted_return_next_session": 0.01,
    }
    row.update(updates)
    return row


def test_delivery_validation_does_not_shift_missing_target_to_a_later_session():
    history = pd.DataFrame(
        {
            "Open": [100.0, 104.0],
            "High": [102.0, 106.0],
            "Low": [99.0, 103.0],
            "Close": [101.0, 105.0],
        },
        index=pd.to_datetime(["2026-10-19", "2026-10-21"]),
    )

    result = validate_delivery_prediction(_delivery_record(), history)

    assert result["validation_status"] == "WAITING"
    assert result["expected_d1_date"] == "2026-10-20"
    assert "not present in official EOD history" in result["validation_reason"]


def test_delivery_validation_scores_the_exact_prediction_target_session():
    history = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 104.0],
            "Low": [99.0, 100.5],
            "Close": [101.0, 103.0],
        },
        index=pd.to_datetime(["2026-10-19", "2026-10-20"]),
    )

    result = validate_delivery_prediction(_delivery_record(), history)

    assert result["validation_status"] == "VALIDATED"
    assert "2026-10-20" in result["actual_d1_date"]
    assert result["actual_d1_close"] == 103.0
    assert result["actual_direction_d1"] == "UP"


def _intraday_record(**updates):
    row = {
        "prediction_id": "intraday-1",
        "symbol": "RELIANCE",
        "mode": "Intraday",
        "prediction_horizon": "30m",
        "model_version": "2026-all-nse-intraday-multifactor-replacement",
        "decision_timestamp": "2026-10-08T10:00:00+05:30",
        "signal_horizon_end": "2026-10-08T10:30:00+05:30",
        "signal_price_reference": 100.0,
        "probability_up_30m": 0.7,
        "signal": "BUY",
    }
    row.update(updates)
    return row


def test_30m_validation_uses_the_first_actual_bar_at_or_after_target():
    idx = pd.DatetimeIndex(
        ["2026-10-08T10:29:00+05:30", "2026-10-08T10:30:00+05:30", "2026-10-08T10:31:00+05:30"]
    )
    bars = pd.DataFrame({"close": [99.0, 102.0, 103.0]}, index=idx)

    result = intraday.validate_one(_intraday_record(), bars)

    assert result["validation_status"] == "VALIDATED"
    assert result["actual_timestamp"] == "2026-10-08T10:30:00+05:30"
    assert abs(result["actual_return_30m"] - 0.02) < 1e-9
    assert result["actual_direction_30m"] == "UP"
    assert result["direction_match_30m"] is True
    assert result["signal_direction_match_30m"] is True


def test_30m_validation_never_scores_a_pre_target_bar():
    bars = pd.DataFrame(
        {"close": [99.0]},
        index=pd.DatetimeIndex(["2026-10-08T10:29:00+05:30"]),
    )

    result = intraday.validate_one(_intraday_record(), bars)

    assert result["validation_status"] == "WAITING"
    assert "not persisted yet" in result["validation_reason"]


def test_30m_validation_rejects_targets_outside_the_cash_session():
    record = _intraday_record(
        decision_timestamp="2026-10-08T15:00:00+05:30",
        signal_horizon_end="2026-10-08T15:30:00+05:30",
    )
    bars = pd.DataFrame(
        {"close": [100.0]},
        index=pd.DatetimeIndex(["2026-10-08T15:00:00+05:30"]),
    )

    result = intraday.validate_one(record, bars)

    assert result["validation_status"] == "OUT_OF_SESSION"
    assert "outside the regular NSE cash session" in result["validation_reason"]


def test_minute_partition_naive_index_is_interpreted_as_ist():
    bars = pd.DataFrame({"Close": [100.0]}, index=pd.DatetimeIndex(["2026-10-08 10:30:00"]))

    result = intraday._normalise_bars(bars)

    assert str(result.index[0]) == "2026-10-08 10:30:00+05:30"
