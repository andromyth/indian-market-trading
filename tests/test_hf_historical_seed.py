import pandas as pd
from hf_historical_seed import five_minute_bars, usable_sessions

def test_aggregates_complete_five_minute_bucket():
    ts = pd.date_range("2026-11-09 03:45:00+00:00", periods=5, freq="min")
    frame = pd.DataFrame({"symbol": ["RELIANCE"]*5, "timestamp": ts,
      "open": [100,101,102,103,104], "high": [101,102,103,104,105],
      "low": [99,100,101,102,103], "close": [100.5,101.5,102.5,103.5,104.5],
      "volume": [1]*5})
    out = five_minute_bars(frame)
    assert len(out) == 1
    assert out.iloc[0]["open"] == 100
    assert out.iloc[0]["high"] == 105
    assert out.iloc[0]["low"] == 99
    assert out.iloc[0]["close"] == 104.5
    assert out.iloc[0]["volume"] == 5
    assert out.iloc[0]["timestamp"] == pd.Timestamp("2026-11-09 03:50:00+00:00")
    assert usable_sessions(out) == 1

def test_drops_incomplete_bucket():
    ts = pd.date_range("2026-11-09 03:45:00+00:00", periods=4, freq="min")
    frame = pd.DataFrame({"symbol": ["SBIN"]*4, "timestamp": ts,
      "open": [10]*4, "high": [11]*4, "low": [9]*4, "close": [10.5]*4, "volume": [1]*4})
    assert five_minute_bars(frame).empty
