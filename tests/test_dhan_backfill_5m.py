from __future__ import annotations

import numpy as np
import pandas as pd

from dhan_backfill_all_nse_5m_history import normalize_dhan_5m, retain_latest_sessions, usable_session_summary

IST = "Asia/Kolkata"


def test_dhan_start_timestamps_become_completed_five_minute_close_timestamps():
    index = pd.date_range("2026-10-30 09:15", periods=3, freq="5min", tz=IST)
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [10.0, np.nan, 30.0],
        },
        index=index,
    )
    bars = normalize_dhan_5m(frame, cutoff=pd.Timestamp("2026-10-30 09:26", tz=IST))
    assert bars.index.tolist() == [
        pd.Timestamp("2026-10-30 09:20", tz=IST),
        pd.Timestamp("2026-10-30 09:25", tz=IST),
    ]
    assert pd.isna(bars["Volume"].iloc[1])


def test_backfill_retains_latest_500_usable_sessions_without_fabricating_bars():
    dates = pd.bdate_range("2024-01-01", periods=505)
    parts = []
    for day_no, day in enumerate(dates):
        index = pd.date_range(
            pd.Timestamp(day).replace(hour=9, minute=20).tz_localize(IST),
            periods=50,
            freq="5min",
        )
        close = np.arange(50, dtype=float) + 100.0 + day_no
        parts.append(pd.DataFrame(
            {
                "Open": close - 0.1,
                "High": close + 0.2,
                "Low": close - 0.2,
                "Close": close,
                "Volume": 1000.0,
            },
            index=index,
        ))
    frame = pd.concat(parts).sort_index()
    assert usable_session_summary(frame)["usable_sessions"] == 505
    retained = retain_latest_sessions(frame, 500)
    assert usable_session_summary(retained)["usable_sessions"] == 500
    assert retained.index.min().date() == dates[5].date()
    assert retained.index.max().date() == dates[-1].date()


def test_empty_dhan_history_is_safe():
    empty = normalize_dhan_5m(pd.DataFrame())
    assert empty.empty
    assert usable_session_summary(empty)["usable_sessions"] == 0
    assert retain_latest_sessions(empty, 500).empty
