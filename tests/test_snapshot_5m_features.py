from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from snapshot_5m_features import (
    FEATURES,
    FEATURE_SCHEMA_VERSION,
    SNAPSHOT_CADENCE_MINUTES,
    latest_snapshot_features,
    make_snapshot_feature_frame,
    resample_minute_candles_to_snapshot_5m,
)

IST = "Asia/Kolkata"


def test_minute_training_candles_align_to_five_minute_snapshot_endpoints():
    index = pd.date_range("2026-10-28 09:15", periods=16, freq="min", tz=IST, name="Datetime")
    close = pd.Series(np.arange(100.0, 116.0), index=index)
    minute = pd.DataFrame({
        "Open": close - 0.25,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": np.arange(1.0, 17.0),
    }, index=index)

    out = resample_minute_candles_to_snapshot_5m(minute)

    assert list(out.index.strftime("%H:%M")) == ["09:20", "09:25", "09:30"]
    assert out.loc[pd.Timestamp("2026-10-28 09:20", tz=IST), "Close"] == pytest.approx(104.0)
    assert out.loc[pd.Timestamp("2026-10-28 09:20", tz=IST), "Volume"] == pytest.approx(15.0)
    assert out.loc[pd.Timestamp("2026-10-28 09:20", tz=IST), "day_open"] == pytest.approx(99.75)
    assert "Open" not in out.columns
    assert "High" not in out.columns
    assert "Low" not in out.columns


def test_snapshot_features_have_five_minute_semantics_and_inference_parity():
    idx = pd.date_range("2026-10-27 09:20", periods=60, freq="5min", tz=IST, name="Datetime")
    close = 100.0 * np.cumprod(1.0 + np.linspace(0.0002, 0.001, len(idx)))
    volume = np.linspace(1000.0, 5000.0, len(idx))
    frame = pd.DataFrame({
        "Close": close,
        "Volume": volume,
        "day_open": 99.5,
        "day_high": close * 1.01,
        "day_low": close * 0.99,
        "day_volume": np.cumsum(volume),
    }, index=idx)

    feature_frame = make_snapshot_feature_frame(frame)
    latest = latest_snapshot_features(frame)

    assert SNAPSHOT_CADENCE_MINUTES == 5
    assert FEATURE_SCHEMA_VERSION == "nse_5m_snapshot_close_v1"
    assert len(FEATURES) == 24
    assert list(feature_frame.columns) == FEATURES
    assert latest["ret_5m"] == pytest.approx(close[-1] / close[-2] - 1.0)
    assert latest["ret_15m"] == pytest.approx(close[-1] / close[-4] - 1.0)
    assert latest["session_return"] == pytest.approx(close[-1] / 99.5 - 1.0)
    assert latest["fno_available"] == 0.0
    for name in FEATURES:
        left, right = feature_frame.iloc[-1][name], latest[name]
        if pd.isna(left):
            assert pd.isna(right)
        else:
            assert right == pytest.approx(float(left))


def test_snapshot_feature_adapter_uses_supplied_day_context_not_fake_candle_ohlc():
    idx = pd.date_range("2026-10-28 09:15", periods=20, freq="5min", tz=IST, name="Datetime")
    close = np.linspace(200.0, 204.0, len(idx))
    frame = pd.DataFrame({
        "Close": close,
        "Volume": np.arange(1.0, 21.0),
        "day_open": 199.0,
        "day_high": 205.0,
        "day_low": 198.5,
    }, index=idx)
    normalized = frame.copy()
    normalized["day_volume"] = np.cumsum(frame["Volume"])

    latest = latest_snapshot_features(normalized, fno={"pcr": 1.2, "oi_change": 200.0})
    assert latest["session_return"] == pytest.approx(close[-1] / 199.0 - 1.0)
    assert latest["day_high_ratio"] == pytest.approx(close[-1] / 205.0 - 1.0)
    assert latest["day_low_ratio"] == pytest.approx(close[-1] / 198.5 - 1.0)
    assert latest["fno_pcr"] == pytest.approx(1.2)
    assert latest["fno_oi_change"] == pytest.approx(200.0)
    assert latest["fno_available"] == 1.0
