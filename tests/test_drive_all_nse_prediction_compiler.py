from __future__ import annotations

import pandas as pd

from drive_all_nse_prediction_compiler import _read_history
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def test_read_history_normalizes_timestamps_and_keeps_last_duplicate(tmp_path):
    path = tmp_path / "history.parquet"
    pd.DataFrame({
        "Timestamp": [
            "2026-11-09 09:20:00",
            "2026-11-09 09:20:00",
            "2026-11-09 09:25:00",
        ],
        "Open": [100, 101, 102],
        "High": [101, 102, 103],
        "Low": [99, 100, 101],
        "Close": [100.5, 101.5, 102.5],
        "Volume": [1000, 1100, 1200],
    }).to_parquet(path, index=False)

    result = _read_history(path)

    assert str(result.index.tz) == "Asia/Kolkata"
    assert len(result) == 2
    assert result.iloc[0]["Close"] == 101.5
    assert result.index[-1] == pd.Timestamp("2026-11-09 09:25:00", tz=IST)


def test_read_history_filters_to_requested_symbol(tmp_path):
    path = tmp_path / "live.parquet"
    pd.DataFrame({
        "symbol": ["AAA", "BBB"],
        "Timestamp": ["2026-11-09 09:20:00", "2026-11-09 09:20:00"],
        "Open": [100, 200],
        "High": [101, 201],
        "Low": [99, 199],
        "Close": [100.5, 200.5],
        "Volume": [1000, 2000],
    }).to_parquet(path, index=False)

    result = _read_history(path, "BBB")

    assert len(result) == 1
    assert result.iloc[0]["Close"] == 200.5
