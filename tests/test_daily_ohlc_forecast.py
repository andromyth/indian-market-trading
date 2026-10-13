import numpy as np
import pandas as pd

from daily_ohlc_forecast import forecast_next_sessions


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


def test_returns_seven_coherent_ohlc_forecasts_and_holdout_metrics():
    result = forecast_next_sessions(_synthetic_ohlcv(), n_sessions=7)

    assert result["status"] == "PASS"
    assert result["history_rows"] >= 160
    assert len(result["forecasts"]) == 7
    assert len(result["validation"]) == 7

    for row in result["forecasts"]:
        assert row["horizon"] in range(1, 8)
        assert all(np.isfinite(row[field]) and row[field] > 0 for field in ("open", "high", "low", "close"))
        assert row["high"] >= max(row["open"], row["close"])
        assert row["low"] <= min(row["open"], row["close"])
        assert row["validation_status"] in {"BEATS_MEDIAN_BASELINE", "LOW_VALIDATION_SKILL"}
        assert row["targets_evaluated"] <= 4


def test_rejects_too_short_daily_history():
    result = forecast_next_sessions(_synthetic_ohlcv(120), n_sessions=7)

    assert result["status"] == "INSUFFICIENT_HISTORY"
    assert result["forecasts"] == []


def test_supports_yfinance_multiindex_columns():
    frame = _synthetic_ohlcv()
    frame.columns = pd.MultiIndex.from_tuples(
        [(name, "TCS.NS") for name in frame.columns]
    )

    result = forecast_next_sessions(frame, n_sessions=3)

    assert result["status"] == "PASS"
    assert len(result["forecasts"]) == 3
