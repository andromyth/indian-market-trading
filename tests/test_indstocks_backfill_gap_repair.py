from __future__ import annotations

import pandas as pd

from indstocks_backfill_all_nse_5m_history import (
    _align_to_monday,
    _checkpoint_contains_partial_sessions,
    _recent_incomplete_session_dates,
)


def _session_frame(day: str, missing: set[str] | None = None, symbol: str = "SBIN") -> pd.DataFrame:
    index = pd.date_range(
        f"{day} 09:20",
        f"{day} 15:30",
        freq="5min",
        tz="Asia/Kolkata",
        name="Datetime",
    )
    missing = missing or set()
    index = index[~index.strftime("%H:%M").isin(missing)]
    return pd.DataFrame(
        {
            "Symbol": symbol,
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 1000,
        },
        index=index,
    )


def test_recent_session_gap_is_flagged_but_complete_sessions_are_not() -> None:
    complete = _session_frame("2026-11-05")
    partial = _session_frame("2026-11-04", {"10:00"})
    frame = pd.concat([partial, complete]).sort_index()

    gaps = _recent_incomplete_session_dates(
        frame,
        pd.Timestamp("2026-11-05 15:30", tz="Asia/Kolkata").to_pydatetime(),
    )

    assert gaps == ["2026-11-04"]


def test_recent_session_gap_detects_a_whole_missing_session_between_observed_days() -> None:
    before = _session_frame("2026-11-03")
    after = _session_frame("2026-11-05")
    frame = pd.concat([before, after]).sort_index()

    gaps = _recent_incomplete_session_dates(
        frame,
        pd.Timestamp("2026-11-06 15:30", tz="Asia/Kolkata").to_pydatetime(),
    )

    assert gaps == ["2026-11-04", "2026-11-06"]


def test_checkpoint_with_74_bars_is_not_reused() -> None:
    partial = _session_frame("2026-11-04", {"10:00"})
    assert len(partial) == 74
    assert _checkpoint_contains_partial_sessions(partial)


def test_checkpoint_with_75_bars_is_reusable() -> None:
    complete = _session_frame("2026-11-05")
    assert len(complete) == 75
    assert not _checkpoint_contains_partial_sessions(complete)


def test_checkpoint_with_an_entire_missing_session_between_observed_days_is_not_reused() -> None:
    before = _session_frame("2026-11-03")
    after = _session_frame("2026-11-05")
    saved = pd.concat([before, after]).sort_index()
    assert _checkpoint_contains_partial_sessions(saved)


def test_checkpoint_window_alignment_is_stable_within_the_same_week() -> None:
    monday = pd.Timestamp("2026-11-02 15:30", tz="Asia/Kolkata").to_pydatetime()
    friday = pd.Timestamp("2026-11-06 15:30", tz="Asia/Kolkata").to_pydatetime()
    assert _align_to_monday(monday) == _align_to_monday(friday)
    assert pd.Timestamp(_align_to_monday(monday)).isoformat() == "2026-11-02T00:00:00+05:30"
