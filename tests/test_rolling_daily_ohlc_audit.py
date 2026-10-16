from __future__ import annotations

import numpy as np
import pandas as pd

from rolling_daily_ohlc_audit import run_rolling_origin_audit


def _synthetic_ohlcv(n: int = 370) -> pd.DataFrame:
    rng = np.random.default_rng(20261016)
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = 250.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, size=n)))
    gap = rng.normal(0.0, 0.004, size=n)
    open_ = np.r_[close[0], close[:-1]] * np.exp(gap)
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.008, size=n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.008, size=n)))
    volume = rng.integers(100_000, 3_000_000, size=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def test_rolling_origin_audit_purges_each_fold_and_reports_baselines():
    frame = _synthetic_ohlcv()
    report, predictions, metrics = run_rolling_origin_audit(
        frame, pd.DatetimeIndex(frame.index), symbol="SYNTH",
        min_train_origins=80, step_sessions=25, max_origins=5,
        min_evaluation_origins=3,
    )
    assert report["status"] == "COMPLETE"
    assert report["safety"]["read_only"] is True
    assert report["safety"]["production_model_changed"] is False
    assert report["safety"]["orders_placed"] is False
    assert report["rolling_origin_count"] >= 3
    # The most recent fully mature target date must be evaluated even if it does
    # not lie on the regular step_sessions grid.
    assert predictions["target_date"].max() == frame.index.max().date().isoformat()
    assert report["latest_evaluated_target_date"] == frame.index.max().date().isoformat()
    assert len(predictions) == report["rolling_origin_count"] * 7
    assert set(predictions["horizon"].unique()) == set(range(1, 8))
    assert all(fold["leakage_check_pass"] for fold in report["folds"])
    for fold in report["folds"]:
        assert fold["max_training_label_date"] < fold["origin_date"]
    assert {"raw_mae_bps", "coherent_mae_bps", "median_baseline_mae_bps",
            "previous_close_baseline_mae_bps"}.issubset(metrics.columns)
    assert metrics.loc[metrics["scope"].eq("per_horizon_target")].shape[0] == 28
    assert report["coherence_corrections"]["origin_horizon_rows"] == len(predictions)
    assert report["aggregate_close_metric"]["target"] == "close"


def test_missing_stock_session_is_never_scored_as_a_target():
    frame = _synthetic_ohlcv()
    calendar = pd.DatetimeIndex(frame.index)
    missing_date = calendar[285]
    incomplete = frame.drop(index=missing_date)
    report, predictions, _ = run_rolling_origin_audit(
        incomplete, calendar, symbol="SYNTH", min_train_origins=80,
        step_sessions=25, max_origins=5, min_evaluation_origins=2,
    )
    assert report["status"] == "COMPLETE"
    assert missing_date.date().isoformat() not in set(predictions["target_date"])
    assert report["missing_stock_sessions_on_calendar"] == 1


def test_short_history_fails_closed_without_predictions():
    frame = _synthetic_ohlcv(120)
    report, predictions, metrics = run_rolling_origin_audit(
        frame, pd.DatetimeIndex(frame.index), symbol="SYNTH", min_train_origins=80,
    )
    assert report["status"] == "BLOCKED_SHORT_SESSION_CALENDAR"
    assert predictions.empty
    assert metrics.empty
