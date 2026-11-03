from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import colab_build_all_nse_intraday_multifactor as intraday_builder
from all_nse_intraday_model import _session_aware_returns
from colab_build_all_nse_intraday_multifactor import (
    make_features,
    make_market_context,
    session_aware_pct_change,
)

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
        for period in (1, 3, 5, 10, 15, 30):
            expected = _session_aware_returns(observed_price, period)
            assert np.isclose(
                features.loc[stamp, f"ret_{period}"], expected, equal_nan=True
            ), f"stock return mismatch period={period} at {stamp}"
            assert np.isclose(
                features.loc[stamp, f"market_ret_{period}"], expected, equal_nan=True
            ), f"market return mismatch period={period} at {stamp}"


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



def test_grouped_training_returns_match_reference_across_sessions_and_missing_values():
    first = pd.date_range("2026-10-26 09:20", periods=8, freq="5min", tz=IST)
    short = pd.date_range("2026-10-27 09:20", periods=2, freq="5min", tz=IST)
    third = pd.date_range("2026-10-28 09:20", periods=7, freq="5min", tz=IST)
    index = first.append(short).append(third)
    values = pd.Series(
        [100, 101, np.nan, 103, 104, 105, 106, 107,
         110, 111, 112, 113, np.nan, 115, 116, 117, 118],
        index=index,
        dtype=float,
    )

    def reference(series, period):
        result = pd.Series(np.nan, index=series.index, dtype=float)
        session_dates = pd.Index(series.index.date)
        previous_session_return = np.nan
        for day in pd.unique(session_dates):
            positions = np.flatnonzero(session_dates == day)
            session = series.iloc[positions].dropna()
            if session.empty:
                previous_session_return = np.nan
                continue
            if len(session) > period:
                returns = session.pct_change(period, fill_method=None)
                if np.isfinite(previous_session_return):
                    returns.iloc[:period] = previous_session_return
                result.loc[session.index] = returns
                previous_session_return = float(returns.iloc[-1])
            else:
                if np.isfinite(previous_session_return):
                    result.loc[session.index] = previous_session_return
                previous_session_return = np.nan
        return result

    for period in (1, 3, 5, 10):
        pd.testing.assert_series_equal(
            session_aware_pct_change(values, period),
            reference(values, period),
        )


def test_precomputed_nifty_context_is_reused_without_recalculation(monkeypatch):
    market = _history()
    stock = market.iloc[::2].copy()
    context = make_market_context(market)

    def unexpected_rebuild(_market):
        pytest.fail("NIFTY context was recalculated for an individual stock")

    monkeypatch.setattr(intraday_builder, "make_market_context", unexpected_rebuild)
    features = make_features(stock, market, market_context=context)
    aligned = context.reindex(
        stock.index, method="ffill", tolerance=pd.Timedelta(minutes=2)
    )

    for column in (
        "market_ret_1", "market_ret_3", "market_ret_5",
        "market_ret_10", "market_ret_15", "market_ret_30", "market_rsi14",
    ):
        pd.testing.assert_series_equal(
            features[column], aligned[column], check_names=False
        )
