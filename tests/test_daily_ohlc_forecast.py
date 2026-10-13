import numpy as np
import pandas as pd

from daily_ohlc_forecast import (
    _cohere_log_return_predictions,
    forecast_next_sessions,
    guard_daily_forecast_for_display,
)


def _synthetic_ohlcv(n: int = 520) -> pd.DataFrame:
    rng = np.random.default_rng(20261014)
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, size=n)))
    gap = rng.normal(0.0, 0.004, size=n)
    open_ = np.r_[close[0], close[:-1]] * np.exp(gap)
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.008, size=n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.008, size=n)))
    volume = rng.integers(100_000, 2_000_000, size=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _forecast_result(as_of: str = "2026-10-12") -> dict:
    return {
        "status": "CALCULATED",
        "as_of_date": as_of,
        "calculation_status": "CALCULATED",
        "data_status": "PASS",
        "forecast_status": "CALCULATED",
        "validation_status": "LOW_VALIDATION_SKILL",
        "trade_eligibility": "NOT_ELIGIBLE_PAPER_TRIAL_REQUIRED",
        "forecasts": [
            {"horizon": 1, "open": 100, "high": 105, "low": 95, "close": 102},
            {"horizon": 2, "open": 102, "high": 107, "low": 97, "close": 103},
        ],
    }


def test_returns_seven_coherent_ohlc_forecasts_and_holdout_metrics():
    result = forecast_next_sessions(_synthetic_ohlcv(), n_sessions=7)

    # Calculation success is intentionally not named PASS or trading approval.
    assert result["status"] == "CALCULATED"
    assert result["calculation_status"] == "CALCULATED"
    assert result["data_status"] == "PASS"
    assert result["forecast_status"] == "CALCULATED"
    assert result["trade_eligibility"] == "NOT_ELIGIBLE_PAPER_TRIAL_REQUIRED"
    assert result["history_rows"] >= 160
    assert len(result["forecasts"]) == 7
    assert len(result["validation"]) == 7

    for row in result["forecasts"]:
        assert row["horizon"] in range(1, 8)
        assert all(
            np.isfinite(row[field]) and row[field] > 0
            for field in ("open", "high", "low", "close")
        )
        assert row["high"] >= max(row["open"], row["close"])
        assert row["low"] <= min(row["open"], row["close"])
        assert row["validation_status"] in {
            "BEATS_MEDIAN_BASELINE",
            "LOW_VALIDATION_SKILL",
        }
        assert row["targets_evaluated"] <= 4

    assert result["validation_status"] in {
        "BASELINE_EDGE_ON_CHRONOLOGICAL_HOLDOUT",
        "LOW_VALIDATION_SKILL",
    }
    # Even an apparently positive one-split holdout must pass a later paper trial.
    assert result["trade_eligibility"] != "TRADE_ELIGIBLE"


def test_rejects_too_short_daily_history():
    result = forecast_next_sessions(_synthetic_ohlcv(120), n_sessions=7)

    assert result["status"] == "INSUFFICIENT_HISTORY"
    assert result["calculation_status"] == "NOT_CALCULATED"
    assert result["trade_eligibility"] == "NOT_ELIGIBLE"
    assert result["forecasts"] == []


def test_supports_yfinance_multiindex_columns():
    frame = _synthetic_ohlcv()
    frame.columns = pd.MultiIndex.from_tuples(
        [(name, "TCS.NS") for name in frame.columns]
    )

    result = forecast_next_sessions(frame, n_sessions=3)

    assert result["status"] == "CALCULATED"
    assert len(result["forecasts"]) == 3


def test_ohlc_coherence_is_applied_to_log_return_forecasts():
    # Deliberately inconsistent raw forecast: high is below open/close and low
    # is above them. The common projection used in both scoring and output fixes it.
    raw = np.log(
        np.array([
            [102.0 / 100.0, 99.0 / 100.0, 101.0 / 100.0, 104.0 / 100.0],
        ])
    )
    projected = _cohere_log_return_predictions(raw, np.array([100.0]))
    prices = 100.0 * np.exp(projected[0])

    assert prices[1] >= max(prices[0], prices[3])
    assert prices[2] <= min(prices[0], prices[3])


def test_stale_daily_source_is_withheld_when_first_target_session_has_started():
    result = guard_daily_forecast_for_display(
        _forecast_result(as_of="2026-10-12"),
        expected_latest_session="2026-10-12",
        now_ist="2026-10-13T16:30:00+05:30",
        target_sessions=[
            "2026-10-13",
            "2026-10-14",
        ],
    )

    assert result["status"] == "STALE_TARGET_DATE"
    assert result["forecast_status"] == "WITHHELD"
    assert result["trade_eligibility"] == "NOT_ELIGIBLE_PAPER_TRIAL_REQUIRED"
    assert result["forecasts"] == []
    assert "already started or passed" in result["reason"]


def test_stale_source_reference_is_withheld_even_before_target_date_check():
    result = guard_daily_forecast_for_display(
        _forecast_result(as_of="2026-10-12"),
        expected_latest_session="2026-10-13",
        now_ist="2026-10-13T18:10:00+05:30",
        target_sessions=["2026-10-14", "2026-10-15"],
    )

    assert result["status"] == "STALE_HISTORY"
    assert result["data_status"] == "STALE"
    assert result["forecasts"] == []


def test_same_day_forecast_is_allowed_only_before_regular_session_open():
    result = guard_daily_forecast_for_display(
        _forecast_result(as_of="2026-10-12"),
        expected_latest_session="2026-10-12",
        now_ist="2026-10-13T08:45:00+05:30",
        target_sessions=["2026-10-13", "2026-10-14"],
    )

    assert result["status"] == "CALCULATED"
    assert result["target_sessions"] == ["2026-10-13", "2026-10-14"]
    assert result["trade_eligibility"] == "NOT_ELIGIBLE_PAPER_TRIAL_REQUIRED"


def test_fresh_post_close_snapshot_can_forecast_the_next_session():
    result = guard_daily_forecast_for_display(
        _forecast_result(as_of="2026-10-13"),
        expected_latest_session="2026-10-13",
        now_ist="2026-10-13T18:10:00+05:30",
        target_sessions=["2026-10-14", "2026-10-15"],
    )

    assert result["status"] == "CALCULATED"
    assert result["target_sessions"][0] == "2026-10-14"
    assert result["trade_eligibility"] == "NOT_ELIGIBLE_PAPER_TRIAL_REQUIRED"


def test_even_positive_validation_is_not_automatic_trade_approval():
    result = _forecast_result()
    result["validation_status"] = "BASELINE_EDGE_ON_CHRONOLOGICAL_HOLDOUT"
    result["validation_gate"] = "PASS_ON_SINGLE_CHRONOLOGICAL_HOLDOUT_ONLY"

    guarded = guard_daily_forecast_for_display(
        result,
        expected_latest_session="2026-10-12",
        now_ist="2026-10-12T08:45:00+05:30",
        target_sessions=["2026-10-12", "2026-10-13"],
    )

    # Current test case is a past-date target (outside pre-open on date 12, but
    # is pre-open, so same-day first target is acceptable for this gate).
    assert guarded["status"] == "CALCULATED"
    assert guarded["trade_eligibility"] == "NOT_ELIGIBLE_PAPER_TRIAL_REQUIRED"
