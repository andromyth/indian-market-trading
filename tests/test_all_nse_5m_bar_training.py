from __future__ import annotations

import numpy as np
import pandas as pd

from colab_build_all_nse_intraday_multifactor import resample_completed_5m_bars

IST = "Asia/Kolkata"


def test_resample_completed_five_minute_bars_aggregates_ohlcv_and_drops_partial():
    index = pd.date_range(
        "2026-10-29 09:15:00",
        periods=11,
        freq="min",
        tz=IST,
    )
    frame = pd.DataFrame(
        {
            "Open": np.arange(100.0, 111.0),
            "High": np.arange(101.0, 112.0),
            "Low": np.arange(99.0, 110.0),
            "Close": np.arange(100.0, 111.0),
            "Volume": np.arange(1.0, 12.0),
        },
        index=index,
    )

    bars = resample_completed_5m_bars(frame)

    assert bars.index.tolist() == [
        pd.Timestamp("2026-10-29 09:20:00", tz=IST),
        pd.Timestamp("2026-10-29 09:25:00", tz=IST),
    ]
    assert bars.iloc[0]["Open"] == 100.0
    assert bars.iloc[0]["High"] == 105.0
    assert bars.iloc[0]["Low"] == 99.0
    assert bars.iloc[0]["Close"] == 104.0
    assert bars.iloc[0]["Volume"] == 15.0
    assert bars.iloc[1]["Close"] == 109.0
    assert bars.iloc[1]["Volume"] == 40.0


def test_resample_does_not_fabricate_an_ohlcv_bar_from_sparse_rows():
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-10-29 09:15:00", tz=IST),
            pd.Timestamp("2026-10-29 09:17:00", tz=IST),
            pd.Timestamp("2026-10-29 09:19:00", tz=IST),
        ]
    )
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [10.0, 10.0, 10.0],
        },
        index=index,
    )

    bars = resample_completed_5m_bars(frame)

    assert bars.empty

def test_fno_join_normalizes_millisecond_and_nanosecond_session_dates():
    from colab_build_all_nse_intraday_multifactor import fno_join_features

    feature_index = pd.DatetimeIndex(
        [pd.Timestamp("2026-10-29 10:00:00", tz=IST)]
    )
    features = pd.DataFrame({"target_30m": [0.01]}, index=feature_index)
    fno = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-10-27", "2026-10-29"]).astype(
                "datetime64[ms]"
            ),
            "Symbol": ["ABC", "ABC"],
            "pcr": [1.25, 9.99],
        }
    )

    joined = fno_join_features(features, fno, "ABC")

    # Same-session F&O data is excluded; the prior session remains available.
    assert joined["fno_pcr"].iloc[0] == 1.25
    assert joined["fno_available"].iloc[0] == 1.0
