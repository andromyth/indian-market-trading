from __future__ import annotations

import pandas as pd
import pytest

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


def test_indstocks_batch_uses_one_five_symbol_request_and_returns_completed_bar_timestamps(monkeypatch):
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
    # Source candle-open timestamps are normalized once to completed bar closes.
    assert frames["RELIANCE"].index[0] == pd.Timestamp("2026-10-30 09:20", tz=IST)
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


def test_instrument_mapping_prefers_eq_but_keeps_symbols_with_other_equity_series(monkeypatch):
    import indstocks_client
    import indstocks_backfill_all_nse_5m_history as backfill

    frame = pd.DataFrame({
        "EXCH": ["NSE", "NSE", "NSE"],
        "SEGMENT": ["E", "E", "E"],
        "SECURITY_ID": ["11", "12", "13"],
        "TRADING_SYMBOL": ["SBIN", "SBIN", "NEWSTOCK"],
        "SERIES": ["BE", "EQ", "BE"],
    })
    monkeypatch.setattr(indstocks_client, "_load_instruments", lambda exchange: frame)
    codes = backfill._instrument_codes()
    assert codes == {"SBIN": "NSE_12", "NEWSTOCK": "NSE_13"}



def test_indstocks_batch_calls_guard_for_each_seven_day_window(monkeypatch):
    import indstocks_client
    import indstocks_backfill_all_nse_5m_history as backfill

    calls = []
    guards = []

    class Response:
        @staticmethod
        def json():
            return {"success": True, "data": {"NSE_100": {"candles": []}}}

    monkeypatch.setattr(indstocks_client, "_request", lambda *args, **kwargs: (calls.append(args) or Response()))
    monkeypatch.setattr(backfill.time, "sleep", lambda _: None)
    symbols = [("RELIANCE", "NSE_100")]
    start = pd.Timestamp("2026-01-01 09:15", tz=IST).to_pydatetime()
    end = pd.Timestamp("2026-01-16 09:15", tz=IST).to_pydatetime()

    _, request_count = fetch_indstocks_5m_batch(
        symbols, start, end, pause_seconds=0,
        request_guard=lambda: guards.append(True),
    )

    assert request_count == 3
    assert len(calls) == 3
    assert len(guards) == 3



def test_indstocks_checkpoint_records_omitted_code_as_empty_window(monkeypatch, tmp_path):
    import json
    import indstocks_client
    import indstocks_backfill_all_nse_5m_history as backfill

    class Response:
        @staticmethod
        def json():
            return {
                "success": True,
                # INDstocks may omit codes with no candles in an old window.
                "data": {
                    "NSE_100": {"candles": [{
                        "ts": int(pd.Timestamp("2026-10-30 09:15", tz=IST).timestamp()),
                        "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 1000,
                    }]},
                },
            }

    class Store:
        def __init__(self):
            self.root_id = "test-root"
            self.uploads = []

        def _find_child(self, parent, name, mime_type=None):
            return None

        def upload_or_update(self, *args, **kwargs):
            self.uploads.append(args)
            return {"status": "OK"}

    store = Store()
    monkeypatch.setattr(indstocks_client, "_request", lambda *args, **kwargs: Response())
    monkeypatch.setattr(backfill.time, "sleep", lambda _: None)
    symbols = [("RELIANCE", "NSE_100"), ("SBIN", "NSE_200")]
    start = pd.Timestamp("2026-10-30 09:15", tz=IST).to_pydatetime()
    end = pd.Timestamp("2026-10-31 09:15", tz=IST).to_pydatetime()

    frames, request_count = fetch_indstocks_5m_batch(
        symbols, start, end, pause_seconds=0,
        store=store, checkpoint_root=tmp_path,
    )

    assert request_count == 1
    assert not frames["RELIANCE"].empty
    assert frames["SBIN"].empty
    assert len(store.uploads) == 2
    manifests = list(tmp_path.rglob("*.parquet.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["provider_omitted_codes"] == ["NSE_200"]
    assert manifest["row_counts"] == {"RELIANCE": 1, "SBIN": 0}


def test_indstocks_batch_retries_transient_ssl_eof_before_succeeding(monkeypatch):
    import indstocks_client
    import indstocks_backfill_all_nse_5m_history as backfill

    attempts = []
    sleeps = []

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
                },
            }

    def flaky_request(*args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError(
                "SSLError: [SSL: UNEXPECTED_EOF_WHILE_READING] unexpected eof while reading"
            )
        return Response()

    monkeypatch.setattr(indstocks_client, "_request", flaky_request)
    monkeypatch.setattr(backfill.time, "sleep", lambda seconds: sleeps.append(seconds))

    start = pd.Timestamp("2026-10-30 09:15", tz=IST).to_pydatetime()
    end = pd.Timestamp("2026-10-31 09:15", tz=IST).to_pydatetime()
    frames, request_count = fetch_indstocks_5m_batch(
        [("RELIANCE", "NSE_100")], start, end, pause_seconds=0,
    )

    assert len(attempts) == 3
    assert sleeps == [2, 4]
    assert request_count == 1
    assert frames["RELIANCE"].iloc[0]["Close"] == 100.5


def test_indstocks_batch_does_not_retry_nontransient_ssl_certificate_error(monkeypatch):
    import indstocks_client
    import indstocks_backfill_all_nse_5m_history as backfill

    attempts = []

    def bad_request(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("SSLError: CERTIFICATE_VERIFY_FAILED")

    monkeypatch.setattr(indstocks_client, "_request", bad_request)
    monkeypatch.setattr(backfill.time, "sleep", lambda _: None)

    start = pd.Timestamp("2026-10-30 09:15", tz=IST).to_pydatetime()
    end = pd.Timestamp("2026-10-31 09:15", tz=IST).to_pydatetime()
    with pytest.raises(RuntimeError, match="CERTIFICATE_VERIFY_FAILED"):
        fetch_indstocks_5m_batch(
            [("RELIANCE", "NSE_100")], start, end, pause_seconds=0,
        )

    assert len(attempts) == 1



def test_indstocks_recent_live_history_reads_only_requested_symbols(monkeypatch, tmp_path):
    import indstocks_backfill_all_nse_5m_history as backfill

    end = pd.Timestamp("2026-11-06 15:30", tz=IST).to_pydatetime()
    monkeypatch.setattr(
        backfill,
        "_resolve_remote_file",
        lambda _store, _path: {"id": "live-folder"},
    )
    monkeypatch.setattr(
        backfill,
        "is_nse_trading_day",
        lambda day: day == end.date(),
        raising=False,
    )

    class Request:
        pass

    class Files:
        def list(self, **_kwargs):
            class Result:
                @staticmethod
                def execute():
                    return {"files": [{"id": "capture-id", "name": "capture_test.parquet"}]}
            return Result()

        @staticmethod
        def get_media(fileId):
            return Request()

    class Service:
        @staticmethod
        def files():
            return Files()

    class Store:
        service = Service()

    class Downloader:
        def __init__(self, output, _request, chunksize):
            self.output = output
            assert chunksize == 8 * 1024 * 1024

        def next_chunk(self):
            self.output.write(b"parquet-placeholder")
            return None, True

    import googleapiclient.http
    monkeypatch.setattr(googleapiclient.http, "MediaIoBaseDownload", Downloader)

    read_calls = []

    def fake_read_parquet(path, *, columns=None, filters=None):
        read_calls.append({"columns": columns, "filters": filters})
        return pd.DataFrame({
            "symbol": ["RELIANCE"],
            "Timestamp": [pd.Timestamp("2026-11-06 09:20", tz=IST)],
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.0],
            "Close": [100.5],
            "Volume": [1000.0],
        })

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    history, inventory = backfill._load_recent_live_history(
        Store(),
        {"RELIANCE"},
        end,
        tmp_path,
        lookback_days=1,
    )

    assert len(read_calls) == 1
    assert read_calls[0]["columns"] == ["symbol", "Timestamp", *backfill.OHLCV]
    assert read_calls[0]["filters"] == [("symbol", "in", ["RELIANCE"])]
    assert len(history["RELIANCE"]) == 1
    assert inventory == {"live_capture_files_read": 1, "live_capture_files_failed": 0}
