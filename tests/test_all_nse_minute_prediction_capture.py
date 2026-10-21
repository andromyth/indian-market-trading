from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import all_nse_minute_prediction_capture as capture

IST = ZoneInfo("Asia/Kolkata")


def test_current_quote_is_actual_close_without_fabricated_minute_ohlc():
    minute = pd.Timestamp("2026-10-21 10:00:00", tz=IST)
    quote = {
        "quote_status": "PASS",
        "actual_observed_price": 102.5,
        "quote_observed_at_ist": (minute + pd.Timedelta(seconds=7)).isoformat(),
        "day_volume": 12000,
    }

    record, price_row, day_volume = capture._current_minute_row(
        symbol="RELIANCE",
        snapshot_minute=minute,
        quote=quote,
        prior_history=pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
        previous_day_volume=None,
        previous_volume_date=None,
        snapshot_date=minute.date(),
    )

    assert record["symbol"] == "RELIANCE"
    assert record["actual_market_value"] == 102.5
    assert record["actual_quote_status"] == "PASS"
    assert record["actual_timestamp_basis"] == "collector_observation_time_not_exchange_tick_timestamp"
    assert price_row is not None
    assert price_row["Close"] == 102.5
    assert np.isnan(price_row["Open"])
    assert np.isnan(price_row["High"])
    assert np.isnan(price_row["Low"])
    assert np.isnan(price_row["Volume"])
    assert record["current_minute_ohlc_complete"] is False
    assert day_volume == 12000


def test_cumulative_volume_delta_uses_previous_same_day_observation():
    minute = pd.Timestamp("2026-10-21 10:01:00", tz=IST)
    prior = pd.DataFrame(
        [{"Open": 100, "High": 101, "Low": 99, "Close": 100, "Volume": 50}],
        index=pd.DatetimeIndex([minute - pd.Timedelta(minutes=1)], name="Datetime"),
    )
    quote = {
        "quote_status": "PASS",
        "actual_observed_price": 101.0,
        "quote_observed_at_ist": (minute + pd.Timedelta(seconds=3)).isoformat(),
        "day_volume": 12050,
    }
    record, price_row, _ = capture._current_minute_row(
        symbol="SBIN",
        snapshot_minute=minute,
        quote=quote,
        prior_history=prior,
        previous_day_volume=12000,
        previous_volume_date=minute.date(),
        snapshot_date=minute.date(),
    )
    assert record["minute_volume_delta"] == 50
    assert price_row["Volume"] == 50


def test_missing_or_nonpositive_quote_never_creates_a_price_row():
    minute = pd.Timestamp("2026-10-21 10:02:00", tz=IST)
    quote = {
        "quote_status": "MISSING_QUOTE",
        "actual_observed_price": None,
        "quote_observed_at_ist": minute.isoformat(),
    }
    record, price_row, _ = capture._current_minute_row(
        symbol="SBIN",
        snapshot_minute=minute,
        quote=quote,
        prior_history=pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
        previous_day_volume=None,
        previous_volume_date=None,
        snapshot_date=minute.date(),
    )
    assert record["actual_market_value"] is None
    assert record["prediction_status"] == "MISSING_CURRENT_QUOTE"
    assert price_row is None


def test_actionable_signal_abstains_when_feature_readiness_is_low():
    assert capture._signal(0.8, 0.85) == ("BUY", "BUY")
    assert capture._signal(0.2, 0.85) == ("SELL", "SELL")
    assert capture._signal(0.8, 0.60) == ("BUY", "HOLD")
    assert capture._signal(0.5, 0.90) == ("HOLD", "HOLD")


def test_parse_candles_keeps_exchange_candle_fields_and_ist_timestamps():
    ts = int(pd.Timestamp("2026-10-21 10:00:00", tz=IST).timestamp())
    frame = capture.AllNSEMinutePredictor._parse_candles([
        {"ts": ts, "o": 100, "h": 102, "l": 99, "c": 101, "v": 12},
        {"ts": ts + 60, "o": 101, "h": 103, "l": 100, "c": 102, "v": 18},
    ])
    assert len(frame) == 2
    assert str(frame.index.tz) == "Asia/Kolkata"
    assert frame.iloc[-1]["Close"] == 102
    assert frame.iloc[-1]["Volume"] == 18
