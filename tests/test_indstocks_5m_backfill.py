from __future__ import annotations

import pandas as pd

from indstocks_backfill_all_nse_5m_history import (
    fetch_indstocks_5m_batch,
    normalize_indstocks_5m,
)

IST = "Asia/Kolkata"


def test_indstocks_candle_open_timestamp_is_shifted_once_to_completed_bar_close():
    index = pd.date_range("2026-10-30 09:15", periods=3, freq="5min", tz=IST)
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [10.0, None, 30.0],
        },
        index=index,
    )
    bars = normalize_indstocks_5m(
        frame,
        cutoff=pd.Timestamp("2026-10-30 09:26", tz=IST),
    )
    assert bars.index.tolist() == [
        pd.Timestamp("2026-10-30 09:20", tz=IST),
        pd.Timestamp("2026-10-30 09:25", tz=IST),
    ]
    assert pd.isna(bars["Volume"].iloc[1])


def test_indstocks_batch_uses_one_five_symbol_request_and_returns_open_timestamps(monkeypatch):
    import indstocks_client
    import indstocks_backfill_all_nse_5m_history as backfill

    calls = []

    class Response:
        @staticmethod
        def json():
            return {
                "success": True,
                "data": {
                    "NSE_100": {"candles": [{
                        "ts": int(pd.Timestamp("2026-10-30 09:15", tz=IST).timestamp()),
                        "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 1000,
                    }]},
                    "NSE_200": {"candles": [{
                        "ts": int(pd.Timestamp("2026-10-30 09:15", tz=IST).timestamp()),
                        "o": 200, "h": 201, "l": 199, "c": 200.5, "v": 2000,
                    }]},
                },
            }

    def fake_request(method, path, *, params=None):
        calls.append((method, path, params))
        return Response()

    monkeypatch.setattr(indstocks_client, "_request", fake_request)
    monkeypatch.setattr(backfill.time, "sleep", lambda _: None)
    symbols = [
        ("RELIANCE", "NSE_100"),
        ("SBIN", "NSE_200"),
    ]
    start = pd.Timestamp("2026-10-30 09:15", tz=IST).to_pydatetime()
    end = pd.Timestamp("2026-10-31 09:15", tz=IST).to_pydatetime()

    frames, request_count = fetch_indstocks_5m_batch(
        symbols, start, end, pause_seconds=0,
    )

    assert request_count == 1
    assert len(calls) == 1
    assert calls[0][0:2] == ("GET", "/market/historical/5minute")
    assert calls[0][2]["scrip-codes"] == "NSE_100,NSE_200"
    # The fetcher preserves source open timestamps; normalization happens once
    # in the archive-writing path.
    assert frames["RELIANCE"].index[0] == pd.Timestamp("2026-10-30 09:15", tz=IST)
    assert frames["SBIN"].iloc[0]["Close"] == 200.5


def test_indstocks_batch_rejects_more_than_five_symbols():
    start = pd.Timestamp("2026-10-30 09:15", tz=IST).to_pydatetime()
    end = pd.Timestamp("2026-10-31 09:15", tz=IST).to_pydatetime()
    symbols = [(f"STOCK{i}", f"NSE_{i}") for i in range(6)]
    try:
        fetch_indstocks_5m_batch(symbols, start, end, pause_seconds=0)
    except ValueError as exc:
        assert "at most 5" in str(exc)
    else:
        raise AssertionError("Expected five-symbol API cap to be enforced")
