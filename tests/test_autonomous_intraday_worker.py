from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from autonomous_intraday_worker import (
    current_session_completed,
    is_fresh_current_session,
    normalize_bars,
    segment_should_stop,
)
from selfdriving.sd_intraday_v2 import _session_rows

IST = ZoneInfo("Asia/Kolkata")


def _bars(start: str, count: int, *, tz=IST) -> pd.DataFrame:
    index = pd.date_range(start, periods=count, freq="min", tz=tz)
    close = 100 + np.arange(count) * 0.03 + np.sin(np.arange(count) / 4) * 0.1
    return pd.DataFrame(
        {
            "Open": close - 0.01,
            "High": close + 0.05,
            "Low": close - 0.05,
            "Close": close,
            "Volume": np.full(count, 1000.0),
        },
        index=index,
    )


def test_segment_worker_stops_at_expected_nse_market_bounds():
    assert segment_should_stop(datetime(2026, 10, 13, 12, 15, tzinfo=IST), "morning")
    assert not segment_should_stop(datetime(2026, 10, 13, 12, 14, tzinfo=IST), "morning")
    assert not segment_should_stop(datetime(2026, 10, 13, 15, 29, tzinfo=IST), "afternoon")
    assert segment_should_stop(datetime(2026, 10, 13, 15, 30, tzinfo=IST), "afternoon")
    assert segment_should_stop(datetime(2026, 10, 17, 10, 0, tzinfo=IST), "once")


def test_normalize_and_freshness_use_current_completed_candles():
    now = pd.Timestamp("2026-10-13 11:00:30", tz=IST)
    raw = _bars("2026-10-13 10:55", 6)
    raw.index.name = "timestamp"
    normalized = normalize_bars(raw)
    completed = current_session_completed(normalized, now)
    fresh, details = is_fresh_current_session(completed, now)

    assert len(completed) == 5
    assert fresh is True
    assert details["status"] == "PASS"
    assert completed.index.max() == pd.Timestamp("2026-10-13 10:59:00", tz=IST)


def test_worker_rejects_stale_candles():
    now = pd.Timestamp("2026-10-13 11:10:00", tz=IST)
    old = _bars("2026-10-13 10:00", 5)
    fresh, details = is_fresh_current_session(old, now)
    assert fresh is False
    assert details["status"] == "STALE"


def test_live_inference_can_emit_minute_anchors_while_training_default_stays_five_minutes():
    prices = _bars("2026-10-13 09:15", 100)
    prices["symbol"] = "HCLTECH"
    market = _bars("2026-10-13 09:15", 100)

    minute_rows = _session_rows(
        prices,
        market,
        (5, 15, 30),
        include_future=False,
        decision_interval_minutes=1,
    )
    default_rows = _session_rows(
        prices,
        market,
        (5, 15, 30),
        include_future=False,
    )

    minute_anchors = {
        pd.Timestamp(row["decision_timestamp"])
        for row in minute_rows
    }
    default_anchors = {
        pd.Timestamp(row["decision_timestamp"])
        for row in default_rows
    }

    assert len(minute_anchors) > len(default_anchors) * 4
    assert len(default_anchors) >= 10
    assert all(ts.second == 0 for ts in minute_anchors)
