from __future__ import annotations

import numpy as np
import pandas as pd

from all_nse_intraday_model import _session_aware_returns


def test_live_rolling_returns_match_training_across_session_open():
    prior_session = pd.date_range(
        "2026-10-29 09:20",
        periods=40,
        freq="5min",
        tz="Asia/Kolkata",
    )
    first_current_bar = pd.DatetimeIndex(
        [pd.Timestamp("2026-10-30 09:20", tz="Asia/Kolkata")]
    )
    index = prior_session.append(first_current_bar)
    closes = pd.Series(
        np.concatenate([np.linspace(100.0, 139.0, len(prior_session)), [160.0]]),
        index=index,
    )

    for period in (1, 3, 5, 10, 15, 30):
        expected = closes.pct_change(period).iloc[-1]
        actual = _session_aware_returns(closes, period)
        assert np.isclose(actual, expected), (
            f"period={period}: live={actual}, training={expected}"
        )


def test_live_rolling_return_is_missing_when_history_is_insufficient():
    one_bar = pd.Series(
        [100.0],
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-10-30 09:20", tz="Asia/Kolkata")]
        ),
    )

    assert np.isnan(_session_aware_returns(one_bar, 1))
