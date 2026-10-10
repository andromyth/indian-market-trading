from __future__ import annotations

import pandas as pd

from selfdriving.sd_v2_panel import merge_beta_fno, merge_beta_news, normalize_delivery_panel


def _panel_with_microsecond_utc_timestamps() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2026-10-05", "2026-10-06", "2026-10-07"]
    ).astype("datetime64[us]")
    timestamps = pd.to_datetime(
        [
            "2026-10-05 10:00:00+00:00",
            "2026-10-06 10:00:00+00:00",
            "2026-10-07 10:00:00+00:00",
        ],
        utc=True,
    ).astype("datetime64[us, UTC]")
    return pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "date": dates,
            "decision_timestamp": timestamps,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        }
    )


def test_normalize_delivery_panel_uses_single_utc_precision():
    panel = normalize_delivery_panel(_panel_with_microsecond_utc_timestamps())
    assert str(panel["decision_timestamp"].dtype) == "datetime64[ns, UTC]"


def test_fno_asof_join_handles_microsecond_source_dates():
    panel = _panel_with_microsecond_utc_timestamps()
    source_dates = pd.to_datetime(
        ["2026-10-02", "2026-10-05", "2026-10-06"]
    ).astype("datetime64[us]")
    fno_daily = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "date": source_dates,
            "fno_fut_close": [100.0, 101.0, 103.0],
            "fno_fut_oi": [1000.0, 1100.0, 1200.0],
            "fno_pcr": [0.9, 1.0, 1.1],
            "fno_iv_change": [0.1, -0.1, 0.2],
            "fno_fut_basis": [1.0, 1.1, 1.2],
        }
    )

    out, audit = merge_beta_fno(panel, fno_daily)

    assert len(out) == len(panel)
    assert str(out["decision_timestamp"].dtype) == "datetime64[ns, UTC]"
    assert audit["status"] == "PASS"


def test_news_feature_join_handles_microsecond_utc_timestamps():
    panel = _panel_with_microsecond_utc_timestamps()
    raw_news = pd.DataFrame(
        {
            "symbol": ["A"],
            "published_at": ["2026-10-05 09:00:00+00:00"],
            "text": ["Company A announces a strong quarterly profit and large new order"],
        }
    )

    out, audit = merge_beta_news(panel, raw_news)

    assert len(out) == len(panel)
    assert str(out["decision_timestamp"].dtype) == "datetime64[ns, UTC]"
    assert audit["status"] == "PASS"
