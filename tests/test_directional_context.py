import numpy as np
import pandas as pd

import directional_context as dc


def _stock_history():
    dates = pd.date_range("2026-08-01", periods=70, freq="D")
    close = np.linspace(100.0, 120.0, len(dates))
    return pd.DataFrame({
        "date": dates,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(len(dates), 100000.0),
    })


def _market_history():
    dates = pd.date_range("2026-08-01", periods=70, freq="D")
    close = np.linspace(10000.0, 10600.0, len(dates))
    return pd.DataFrame({
        "Open": close - 5.0,
        "High": close + 10.0,
        "Low": close - 10.0,
        "Close": close,
        "Volume": np.full(len(dates), 1000000.0),
    }, index=dates)


def test_causal_context_filters_future_news(monkeypatch):
    monkeypatch.setattr(
        dc,
        "historical_news",
        lambda symbol, when: [
            {"title": "Strong order win", "summary": "", "published": "2026-08-12 10:00:00+05:30", "source": "Source A"},
            {"title": "Profit warning", "summary": "", "published": "2026-08-13 10:00:00+05:30", "source": "Source B"},
        ],
    )
    monkeypatch.setattr(dc, "historical_announcements", lambda symbol, when: [])

    result = dc.analyze_directional_context(
        "SBIN",
        pd.Timestamp("2026-08-12 15:30", tz="Asia/Kolkata"),
        _stock_history(),
        _market_history(),
        model_p_up=0.70,
        model_expected_return=0.01,
    )

    assert result["status"] == "OK"
    assert result["model_p_up"] == 0.70
    assert result["model_direction"] == "UP"
    assert result["fused_direction"] in {"UP", "NEUTRAL"}
    assert result["news_source_count"] == 1


def test_context_exposes_official_nse_source(monkeypatch):
    monkeypatch.setattr(dc, "historical_news", lambda symbol, when: [])
    monkeypatch.setattr(
        dc,
        "historical_announcements",
        lambda symbol, when: [{"subject": "Dividend", "date": "2026-08-12", "link": ""}],
    )

    result = dc.analyze_directional_context(
        "SBIN",
        pd.Timestamp("2026-08-12 15:30", tz="Asia/Kolkata"),
        _stock_history(),
        _market_history(),
        model_p_up=0.30,
        model_expected_return=-0.01,
    )

    assert result["official_nse_announcement_count"] == 1
    assert result["high_impact_news_count"] >= 1
    assert result["model_direction"] == "DOWN"
