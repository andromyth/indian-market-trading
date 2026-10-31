from __future__ import annotations

import numpy as np
import pandas as pd

from five_minute_history_archive import build_history_archive_frame, merge_history_archive_frames

IST = "Asia/Kolkata"


def _bars(symbol_price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2026-10-30 09:20", periods=3, freq="5min", tz=IST)
    return pd.DataFrame(
        {
            "Open": [symbol_price, symbol_price + 1, symbol_price + 2],
            "High": [symbol_price + 1, symbol_price + 2, symbol_price + 3],
            "Low": [symbol_price - 1, symbol_price, symbol_price + 1],
            "Close": [symbol_price + 0.5, symbol_price + 1.5, symbol_price + 2.5],
            "Volume": [1000.0, np.nan, 1200.0],
        },
        index=idx,
    )


def test_history_archive_keeps_real_five_minute_close_timestamps_and_missing_volume():
    archive = build_history_archive_frame(
        {"RELIANCE": _bars()},
        {"RELIANCE": "INDSTOCKS_HISTORICAL_5M"},
        cutoff=pd.Timestamp("2026-10-30 09:25", tz=IST),
    )
    assert len(archive) == 2
    assert archive["symbol"].unique().tolist() == ["RELIANCE"]
    assert archive["Timestamp"].tolist() == list(_bars().index[:2])
    assert pd.isna(archive.loc[1, "Volume"])
    assert archive["source"].unique().tolist() == ["INDSTOCKS_HISTORICAL_5M"]


def test_history_archive_merge_is_timestamp_idempotent_and_latest_wins():
    first = build_history_archive_frame({"SBIN": _bars(100.0)})
    changed = _bars(100.0).iloc[-1:].copy()
    changed["Close"] = 999.0
    second = build_history_archive_frame({"SBIN": changed})
    merged = merge_history_archive_frames(first, second)
    assert len(merged) == 3
    assert merged.iloc[-1]["Close"] == 999.0


def test_history_archive_empty_input_is_safe():
    assert build_history_archive_frame({}).empty
    assert merge_history_archive_frames(pd.DataFrame(), pd.DataFrame()).empty
