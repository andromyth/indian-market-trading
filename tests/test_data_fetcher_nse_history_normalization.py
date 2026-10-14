import pandas as pd

from data_fetcher import _normalize_historical_frame


def test_normalizer_accepts_official_nse_security_history_schema():
    dates = pd.to_datetime(["2026-09-01", "2026-09-02"])
    raw = pd.DataFrame(
        {
            "Symbol": ["SBIN", "SBIN"],
            "Series": ["EQ", "EQ"],
            "Prev Close": [1060.0, 1034.5],
            "Open Price": [1051.3, 1026.0],
            "High Price": [1051.3, 1035.3],
            "Low Price": [1027.1, 1017.3],
            # Deliberately differ so the test proves Close Price takes precedence.
            "Last Price": [1034.0, 1020.0],
            "Close Price": [1034.5, 1020.9],
            "Total Traded Quantity": [7605132, 13843678],
        },
        index=dates,
    )

    result = _normalize_historical_frame(raw)

    assert len(result) == 2
    assert result["Open"].tolist() == [1051.3, 1026.0]
    assert result["High"].tolist() == [1051.3, 1035.3]
    assert result["Low"].tolist() == [1027.1, 1017.3]
    assert result["Close"].tolist() == [1034.5, 1020.9]
    assert result["Volume"].tolist() == [7605132, 13843678]


def test_normalizer_still_uses_last_price_when_close_price_absent():
    raw = pd.DataFrame(
        {
            "Open Price": [100.0],
            "High Price": [102.0],
            "Low Price": [99.0],
            "Last Price": [101.0],
            "Total Traded Quantity": [123456],
        },
        index=pd.to_datetime(["2026-09-01"]),
    )

    result = _normalize_historical_frame(raw)

    assert len(result) == 1
    assert result["Close"].iloc[0] == 101.0
    assert result["Volume"].iloc[0] == 123456
