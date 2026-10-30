from __future__ import annotations

import numpy as np
import pandas as pd

from all_nse_intraday_model import _session_aware_returns
from colab_build_all_nse_intraday_multifactor import make_features, session_aware_pct_change

IST = "Asia/Kolkata"


def _history() -> pd.DataFrame:
    prior = pd.date_range("2026-10-29 09:20", periods=40, freq="5min", tz=IST)
    current = pd.date_range("2026-10-30 09:20", periods=9, freq="5min", tz=IST)
    index = prior.append(current)
    close = np.concatenate([np.linspace(100.0, 139.0, len(prior)), np.linspace(160.0, 168.0, len(current))])
    return pd.DataFrame(
        {
            "Open": close - 0.2,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.arange(len(index), dtype=float) + 1000,
        },
        index=index,
    )


def test_training_stock_and_market_returns_match_live_session_aware_features():
    prices = _history()
    features = make_features(prices, prices)
    for stamp in prices.index[-9:]:
        observed_price = prices.loc[:stamp, "Close"]
        expected_market = _session_aware_returns(observed_price, 1)
        for period in (1, 3, 5, 10, 15, 30):
            expected = _session_aware_returns(observed_price, period)
            assert np.isclose(
                features.loc[stamp, f"ret_{period}"], expected, equal_nan=True
            ), f"stock return mismatch period={period} at {stamp}"
            assert np.isclose(
                features.loc[stamp, f"market_ret_{period}"], expected, equal_nan=True
            ), f"market return mismatch period={period} at {stamp}"
        assert np.isfinite(expected_market) or np.isnan(expected_market)


def test_training_returns_do_not_include_overnight_gap():
    prices = _history()
    features = make_features(prices, prices)
    first_current_bar = prices.index[-9]
    for period in (1, 3, 5, 10, 15, 30):
        actual = features.loc[first_current_bar, f"ret_{period}"]
        expected = _session_aware_returns(prices.loc[:first_current_bar, "Close"], period)
        assert np.isclose(actual, expected, equal_nan=True)
    assert not np.isclose(
        features.loc[first_current_bar, "ret_1"],
        prices["Close"].pct_change(1).loc[first_current_bar],
    )


def test_session_aware_training_return_handles_empty_history():
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz=IST))
    assert session_aware_pct_change(empty, 1).empty
