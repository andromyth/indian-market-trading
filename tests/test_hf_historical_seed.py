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
    assert usable_sessions(out) == 0  # A single bar is not a complete session

def test_drops_incomplete_bucket():
    ts = pd.date_range("2026-11-09 03:45:00+00:00", periods=4, freq="min")
    frame = pd.DataFrame({"symbol": ["SBIN"]*4, "timestamp": ts,
      "open": [10]*4, "high": [11]*4, "low": [9]*4, "close": [10.5]*4, "volume": [1]*4})
    assert five_minute_bars(frame).empty

 
def test_counts_only_full_regular_sessions():
    from hf_historical_seed import session_quality
    labels = pd.date_range("2026-11-09 09:20:00", "2026-11-09 15:30:00",
                           freq="5min", tz="Asia/Kolkata")
    full = pd.DataFrame({"timestamp": labels.tz_convert("UTC")})
    quality = session_quality(full)
    assert quality["complete_sessions"] == 1
    assert quality["partial_sessions"] == 0
    quality = session_quality(full.iloc[:-1])
    assert quality["complete_sessions"] == 0
    assert quality["partial_sessions"] == 1


def test_preserves_open_interest_as_last_value():
    ts = pd.date_range("2026-11-09 03:45:00+00:00", periods=5, freq="min")
    frame = pd.DataFrame({"symbol": ["NIFTY"] * 5, "timestamp": ts,
      "open": [100] * 5, "high": [101] * 5, "low": [99] * 5,
      "close": [100] * 5, "volume": [1] * 5, "oi": [10, 11, 12, 13, 14]})
    out = five_minute_bars(frame)
    assert len(out) == 1
    assert out.iloc[0]["oi"] == 14
