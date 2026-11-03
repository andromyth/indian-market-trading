from __future__ import annotations

import pandas as pd
import pytest

from ingest_kaggle_nifty_5m_context import prepare_context_frame


def _session(day: str, count: int = 75) -> pd.DataFrame:
    timestamps = pd.date_range(
        f"{day} 09:15:00",
        periods=count,
        freq="5min",
    )
    close = [100.0 + i for i in range(count)]
    return pd.DataFrame({
        "Date": timestamps,
        "Open": close,
        "High": [value + 1 for value in close],
        "Low": [value - 1 for value in close],
        "Close": close,
        "Volume": 0,
    })


def test_context_is_close_only_complete_sessions_and_close_timestamped():
    raw = pd.concat([
        _session("2026-01-05"),
        _session("2026-01-06", count=74),
    ], ignore_index=True)

    output, stats = prepare_context_frame(raw)

    assert len(output) == 74
    assert output["Timestamp"].iloc[0] == pd.Timestamp("2026-01-05 09:25:00")
    assert output["Timestamp"].iloc[-1] == pd.Timestamp("2026-01-05 15:30:00")
    assert output["Close"].iloc[0] == 101.0
    assert output[["Open", "High", "Low", "Volume"]].isna().all().all()
    assert stats["accepted_complete_sessions"] == 1
    assert stats["excluded_or_incomplete_sessions"] == 1
    assert stats["excluded_opening_bars"] == 1


def test_duplicate_timestamps_are_rejected():
    raw = _session("2026-01-05")
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        prepare_context_frame(raw)


def test_separate_date_and_time_columns_are_supported():
    raw = _session("2026-01-05").rename(columns={"Date": "DateTime"})
    raw["date"] = raw["DateTime"].dt.strftime("%Y-%m-%d")
    raw["time"] = raw["DateTime"].dt.strftime("%H:%M:%S")
    raw = raw.drop(columns=["DateTime"])

    output, stats = prepare_context_frame(raw)

    assert len(output) == 74
    assert stats["accepted_complete_sessions"] == 1
