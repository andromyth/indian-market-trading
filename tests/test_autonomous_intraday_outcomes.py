from __future__ import annotations

import pandas as pd
import pytest

from audit_autonomous_intraday_outcomes import evaluate_record, _metrics


def _bars(values):
    idx = pd.DatetimeIndex(pd.to_datetime([
        "2026-10-14T03:45:00Z",
        "2026-10-14T03:50:00Z",
        "2026-10-14T03:51:00Z",
    ], utc=True))
    return pd.DataFrame({"close": values}, index=idx)


def test_evaluate_buy_at_or_after_horizon_target():
    record = {
        "symbol": "DMART.NS",
        "decision_timestamp": "2026-10-14T03:45:00Z",
        "horizon_min": 5,
        "signal": "BUY",
        "probability_up": 0.7,
        "reference_close": 100.0,
        "predicted_price": 101.0,
    }
    row = evaluate_record(record, _bars([100.0, 102.0, 103.0]))
    assert row["validation_status"] == "VALIDATED"
    assert row["actual_price"] == 102.0
    assert row["actual_return"] == pytest.approx(0.02)
    assert row["direction_match"] is True
    assert row["brier_score"] == pytest.approx(0.09)


def test_never_use_a_pre_target_bar():
    record = {
        "symbol": "DMART",
        "decision_timestamp": "2026-10-14T03:45:00Z",
        "horizon_min": 5,
        "signal": "BUY",
        "probability_up": 0.7,
        "reference_close": 100.0,
        "predicted_price": 101.0,
    }
    bars = pd.DataFrame(
        {"close": [100.0]},
        index=pd.DatetimeIndex(pd.to_datetime(["2026-10-14T03:49:00Z"], utc=True)),
    )
    row = evaluate_record(record, bars)
    assert row["validation_status"] == "WAITING_FOR_TARGET_BAR"
    assert pd.isna(row["actual_return"])


def test_metrics_only_score_matured_predictions():
    rows = pd.DataFrame([
        {"validation_status": "VALIDATED", "signal": "BUY", "direction_match": True, "brier_score": 0.04,
         "predicted_price_abs_error": 1.0, "baseline_price_abs_error": 2.0,
         "signed_signal_return_before_costs": 0.01},
        {"validation_status": "WAITING_FOR_TARGET_BAR", "signal": "SELL", "direction_match": None,
         "brier_score": float("nan"), "predicted_price_abs_error": float("nan"),
         "baseline_price_abs_error": float("nan"), "signed_signal_return_before_costs": float("nan")},
    ])
    report = _metrics(rows)
    assert report["predictions_seen"] == 2
    assert report["validated"] == 1
    assert report["action_direction_accuracy"] == 1.0
    assert report["model_beats_reference_price_baseline"] is True
