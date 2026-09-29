import sys
import types

# Keep this unit test runnable in a minimal environment.
feedparser = types.SimpleNamespace(parse=lambda *args, **kwargs: types.SimpleNamespace(entries=[]))
sys.modules.setdefault('feedparser', feedparser)

import numpy as np
import pandas as pd

from data_fetcher import _normalize_historical_frame, _yahoo_chart_history
import data_fetcher


def test_multiindex_field_ticker_shape():
    idx = pd.date_range('2026-09-01 09:15', periods=5, freq='min')
    columns = pd.MultiIndex.from_product([['RELIANCE.NS'], ['Open', 'High', 'Low', 'Close', 'Volume']])
    values = np.array([
        [100, 101, 99, 100.5, 1000],
        [100.5, 102, 100, 101.5, 1100],
        [101.5, 103, 101, 102.5, 1200],
        [102.5, 104, 102, 103.5, 1300],
        [103.5, 105, 103, 104.5, 1400],
    ])
    out = _normalize_historical_frame(pd.DataFrame(values, index=idx, columns=columns))
    assert {'Open', 'High', 'Low', 'Close', 'Volume'} <= set(out.columns)
    assert out['Close'].notna().all()


def test_multiindex_ticker_field_shape():
    idx = pd.date_range('2026-09-01 09:15', periods=3, freq='min')
    columns = pd.MultiIndex.from_product([['RELIANCE.NS'], ['Open', 'High', 'Low', 'Close']])
    values = np.array([[100, 101, 99, 100.5], [100.5, 102, 100, 101.5], [101.5, 103, 101, 102.5]])
    # Reverse the levels to emulate the alternate Yahoo shape: (field, ticker).
    columns = columns.swaplevel(0, 1)
    out = _normalize_historical_frame(pd.DataFrame(values, index=idx, columns=columns))
    assert {'Open', 'High', 'Low', 'Close', 'Volume'} <= set(out.columns)
    assert out['Close'].notna().all()


def test_flat_lowercase_shape():
    idx = pd.date_range('2026-09-01', periods=3)
    df = pd.DataFrame({'open': [1, 2, 3], 'high': [2, 3, 4], 'low': [0, 1, 2], 'close': [1.5, 2.5, 3.5]}, index=idx)
    out = _normalize_historical_frame(df)
    assert {'Open', 'High', 'Low', 'Close', 'Volume'} <= set(out.columns)


def test_bad_shape_returns_empty_frame():
    assert _normalize_historical_frame(pd.DataFrame({'foo': [1, 2]})).empty


class _Response:
    def raise_for_status(self):
        return None
    def json(self):
        return {
            "chart": {"result": [{
                "timestamp": [1788250500, 1788250560],
                "indicators": {
                    "quote": [{
                        "open": [100, 101], "high": [102, 103],
                        "low": [99, 100], "close": [101, 102],
                        "volume": [1000, 1100],
                    }]
                }
            }]}
        }

def test_yahoo_chart_adapter(monkeypatch):
    monkeypatch.setattr(data_fetcher.requests, "get", lambda *args, **kwargs: _Response())
    out = _yahoo_chart_history("BHARTIARTL.NS", "5d", "1d")
    assert not out.empty
    assert list(out[["Open", "High", "Low", "Close", "Volume"]].columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert float(out["Close"].iloc[-1]) == 102.0

print("yahoo chart adapter test: PASS")
