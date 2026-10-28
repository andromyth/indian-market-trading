from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import all_nse_minute_prediction_capture as capture

IST = ZoneInfo("Asia/Kolkata")


def test_session_aware_return_does_not_use_overnight_gap():
    import all_nse_intraday_model as model

    idx = pd.DatetimeIndex([
        pd.Timestamp("2026-10-23 15:28:00", tz=IST),
        pd.Timestamp("2026-10-23 15:29:00", tz=IST),
        pd.Timestamp("2026-10-26 09:15:00", tz=IST),
    ])
    close = pd.Series([99.0, 100.0, 150.0], index=idx)
    assert model._session_aware_returns(close, 1) == pytest.approx(100.0 / 99.0 - 1.0)


def test_current_quote_is_actual_close_without_fabricated_minute_ohlc():
    minute = pd.Timestamp("2026-10-21 10:00:00", tz=IST)
    quote = {
        "quote_status": "PASS",
        "actual_observed_price": 102.5,
        "quote_observed_at_ist": (minute + pd.Timedelta(seconds=7)).isoformat(),
        "day_volume": 12000,
    }

    record, price_row, day_volume = capture._current_minute_row(
        symbol="RELIANCE",
        snapshot_minute=minute,
        quote=quote,
        prior_history=pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
        previous_day_volume=None,
        previous_volume_date=None,
        snapshot_date=minute.date(),
    )

    assert record["symbol"] == "RELIANCE"
    assert record["actual_market_value"] == 102.5
    assert record["actual_quote_status"] == "PASS"
    assert record["actual_quote_age_seconds"] == pytest.approx(7.0)
    assert record["actual_timestamp_basis"] == "collector_observation_time_not_exchange_tick_timestamp"
    assert price_row is not None
    assert price_row["Close"] == 102.5
    assert np.isnan(price_row["Open"])
    assert np.isnan(price_row["High"])
    assert np.isnan(price_row["Low"])
    assert np.isnan(price_row["Volume"])
    assert record["current_minute_ohlc_complete"] is False
    assert day_volume == 12000


def test_cumulative_volume_delta_uses_previous_same_day_observation():
    minute = pd.Timestamp("2026-10-21 10:01:00", tz=IST)
    prior = pd.DataFrame(
        [{"Open": 100, "High": 101, "Low": 99, "Close": 100, "Volume": 50}],
        index=pd.DatetimeIndex([minute - pd.Timedelta(minutes=1)], name="Datetime"),
    )
    quote = {
        "quote_status": "PASS",
        "actual_observed_price": 101.0,
        "quote_observed_at_ist": (minute + pd.Timedelta(seconds=3)).isoformat(),
        "day_volume": 12050,
    }
    record, price_row, _ = capture._current_minute_row(
        symbol="SBIN",
        snapshot_minute=minute,
        quote=quote,
        prior_history=prior,
        previous_day_volume=12000,
        previous_volume_date=minute.date(),
        snapshot_date=minute.date(),
    )
    assert record["minute_volume_delta"] == 50
    assert price_row["Volume"] == 50


def test_missing_or_nonpositive_quote_never_creates_a_price_row():
    minute = pd.Timestamp("2026-10-21 10:02:00", tz=IST)
    quote = {
        "quote_status": "MISSING_QUOTE",
        "actual_observed_price": None,
        "quote_observed_at_ist": minute.isoformat(),
    }
    record, price_row, _ = capture._current_minute_row(
        symbol="SBIN",
        snapshot_minute=minute,
        quote=quote,
        prior_history=pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
        previous_day_volume=None,
        previous_volume_date=None,
        snapshot_date=minute.date(),
    )
    assert record["actual_market_value"] is None
    assert record["prediction_status"] == "MISSING_CURRENT_QUOTE"
    assert price_row is None


def test_actionable_signal_abstains_when_feature_readiness_is_low():
    assert capture._signal(0.8, 0.85) == ("BUY", "BUY")
    assert capture._signal(0.2, 0.85) == ("SELL", "SELL")
    assert capture._signal(0.8, 0.60) == ("BUY", "HOLD")
    assert capture._signal(0.5, 0.90) == ("HOLD", "HOLD")


def test_active_calibration_never_overrides_data_readiness_hold():
    guard = capture._primary_signal_with_readiness_guard
    assert guard("PASS", False, "BUY") == "BUY"
    assert guard("PASS", False, "SELL") == "SELL"
    assert guard("PASS_DATA_LIMITED", True, "BUY") == "HOLD"
    assert guard("PASS", True, "BUY") == "HOLD"
    assert guard("PASS_UNTRAINED_SYMBOL", False, "SELL") == "HOLD"
    assert guard("PASS", None, "BUY") == "HOLD"
    assert guard("PASS", False, "ABSTAIN") == "HOLD"


def test_parse_candles_keeps_exchange_candle_fields_and_ist_timestamps():
    ts = int(pd.Timestamp("2026-10-21 10:00:00", tz=IST).timestamp())
    frame = capture.AllNSEMinutePredictor._parse_candles([
        {"ts": ts, "o": 100, "h": 102, "l": 99, "c": 101, "v": 12},
        {"ts": ts + 60, "o": 101, "h": 103, "l": 100, "c": 102, "v": 18},
    ])
    assert len(frame) == 2
    assert str(frame.index.tz) == "Asia/Kolkata"
    assert frame.iloc[-1]["Close"] == 102
    assert frame.iloc[-1]["Volume"] == 18


def test_run_for_capture_writes_one_actual_and_prediction_row_per_symbol(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from all_nse_minute_prediction_capture import AllNSEMinutePredictor

    minute = pd.Timestamp.now(tz=IST).floor("min")
    predictor = AllNSEMinutePredictor(tmp_path)
    predictor.start = lambda: None
    predictor._seed_ready = True
    predictor._seed_status = "READY"
    predictor._symbols = ["AAA", "BBB"]
    predictor._codes = {"AAA": "NSE_1", "BBB": "NSE_2"}
    predictor._seed_coverage = 2
    predictor._seed_errors = 0
    predictor._trained_symbols = {"AAA", "BBB"}
    predictor._model_status = {
        "training_symbol_count": 2,
        "trained_until": "2026-10-20T09:15:00+00:00",
        "manifest_sha256_match": True,
    }
    predictor._refresh_current_session_history = lambda _minute: {
        "status": "OK", "batch_count": 1, "failed_batches": 0,
        "failed_symbols": [], "updated_symbols": 2, "elapsed_seconds": 0.1,
    }

    idx = pd.date_range(
        end=minute - pd.Timedelta(minutes=1),
        periods=60,
        freq="min",
        tz=IST,
        name="Datetime",
    )
    history = pd.DataFrame(
        {
            "Open": np.linspace(99, 100, len(idx)),
            "High": np.linspace(100, 101, len(idx)),
            "Low": np.linspace(98, 99, len(idx)),
            "Close": np.linspace(99.5, 100.5, len(idx)),
            "Volume": np.full(len(idx), 10.0),
        },
        index=idx,
    )
    predictor._history = {"AAA": history.copy(), "BBB": history.copy()}
    predictor._market_history = history.copy()
    predictor._model_api = SimpleNamespace(
        FEATURES=["f1", "f2"],
        _latest_features=lambda price_df, market_df=None, decision_timestamp=None: {
            "f1": 1.0,
            "f2": 0.5,
        },
    )
    predictor._classifier = SimpleNamespace(
        predict_proba=lambda frame: np.array([[0.2, 0.8], [0.8, 0.2]])
    )
    predictor._regressor = SimpleNamespace(
        predict=lambda frame: np.array([0.01, -0.01])
    )
    predictor._fetch_current_nifty_quote = lambda: (25000.0, minute, None)

    snapshot_dir = tmp_path / "all_universe_minute_snapshots"
    snapshot_dir.mkdir(parents=True)
    quote_path = snapshot_dir / f"snapshot_{minute:%Y%m%d_%H%M}.parquet"
    pd.DataFrame([
        {
            "symbol": "AAA",
            "quote_status": "PASS",
            "actual_observed_price": 101.25,
            "quote_observed_at_ist": (minute + pd.Timedelta(seconds=2)).isoformat(),
            "day_volume": 10000,
        },
        {
            "symbol": "BBB",
            "quote_status": "PASS",
            "actual_observed_price": 202.5,
            "quote_observed_at_ist": (minute + pd.Timedelta(seconds=3)).isoformat(),
            "day_volume": 20000,
        },
    ]).to_parquet(quote_path, index=False)

    class FakeDriveStore:
        def upload_or_update(self, local_path, remote_relative_path, mime_type="application/octet-stream"):
            return {
                "status": "OK",
                "action": "created",
                "remote_path": remote_relative_path,
                "file_id": "fake-drive-id",
            }

    monkeypatch.setattr("persistent_store.GoogleDriveStore", FakeDriveStore)

    result = predictor.run_for_capture(
        {"cycle_minute_ist": minute.isoformat(), "status": "OK"},
        worker_root=tmp_path,
    )

    assert result["status"] == "OK"
    assert result["row_count"] == 2
    assert result["prediction_count"] == 2
    assert result["quality_gate_pass"] is True

    written = pd.read_parquet(
        tmp_path / "all_nse_minute_predictions" / f"{minute:%Y}" / f"{minute:%m}" /
        f"{minute:%d}" / f"predictions_{minute:%Y%m%d_%H%M}.parquet"
    ).set_index("symbol")
    assert written.loc["AAA", "actual_market_value"] == 101.25
    assert written.loc["AAA", "probability_up_30m"] == pytest.approx(0.8)
    assert written.loc["AAA", "signal"] == "BUY"
    assert written.loc["BBB", "actual_market_value"] == 202.5
    assert written.loc["BBB", "probability_up_30m"] == pytest.approx(0.2)
    assert written.loc["BBB", "signal"] == "SELL"

    # The version and readiness segment must be persisted with every prediction,
    # otherwise historical outcomes cannot be safely attributed to this artifact.
    expected_base_version = predictor._production_model_version()
    assert written.loc["AAA", "base_model_version"] == expected_base_version
    assert written.loc["BBB", "base_model_version"] == expected_base_version
    assert written.loc["AAA", "model_version"] == f"{expected_base_version}__READINESS_READY"
    assert written.loc["BBB", "model_version"] == f"{expected_base_version}__READINESS_READY"

    import json
    minute_manifest_path = (
        tmp_path / "all_nse_minute_predictions" / f"{minute:%Y}" / f"{minute:%m}" /
        f"{minute:%d}" / f"manifest_{minute:%Y%m%d_%H%M}.json"
    )
    saved_manifest = json.loads(minute_manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest["model_version"] == expected_base_version
    assert saved_manifest["feature_quality_diagnostics"]["mean_feature_coverage_pct"] == 100.0
    assert saved_manifest["feature_quality_diagnostics"]["feature_group_coverage_pct"]["price_technical"] is None



def test_missing_nifty_context_does_not_break_timestamp_filtering():
    minute = pd.Timestamp("2026-10-27 10:00:00", tz=IST)
    # Empty frames from pandas default to RangeIndex; comparing that index to
    # a Timestamp previously raised TypeError for almost every symbol.
    empty_range_index = pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"]
    )

    empty_market = capture.AllNSEMinutePredictor._market_history_before(
        empty_range_index,
        minute,
    )

    assert empty_market.empty
    assert isinstance(empty_market.index, pd.DatetimeIndex)
    assert str(empty_market.index.tz) == "Asia/Kolkata"

    index = pd.date_range(
        minute - pd.Timedelta(minutes=1),
        periods=3,
        freq="min",
        tz=IST,
        name="Datetime",
    )
    market = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [10.0, 12.0, 11.0],
        },
        index=index,
    )
    before = capture.AllNSEMinutePredictor._market_history_before(market, minute)
    assert len(before) == 1
    assert before.index.max() == minute - pd.Timedelta(minutes=1)


def test_stale_market_history_is_excluded_from_feature_context():
    minute = pd.Timestamp("2026-10-27 10:00:00", tz=IST)
    index = pd.date_range(
        minute - pd.Timedelta(minutes=8),
        periods=3,
        freq="min",
        tz=IST,
        name="Datetime",
    )
    market = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [10.0, 12.0, 11.0],
        },
        index=index,
    )

    before = capture.AllNSEMinutePredictor._market_history_before(market, minute)

    assert before.empty
    assert isinstance(before.index, pd.DatetimeIndex)
    assert str(before.index.tz) == "Asia/Kolkata"


def test_yahoo_nifty_history_fallback_uses_only_real_predecision_candles(monkeypatch):
    import data_fetcher

    minute = pd.Timestamp("2026-10-27 10:00:00", tz=IST)
    idx = pd.date_range(
        minute.tz_convert("UTC") - pd.Timedelta(minutes=2),
        periods=5,
        freq="min",
        tz="UTC",
        name="Datetime",
    )
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [10.0, 12.0, 11.0, 13.0, 14.0],
        },
        index=idx,
    )
    monkeypatch.setattr(
        data_fetcher,
        "_yahoo_chart_history",
        lambda symbol, period, interval: frame,
    )

    result = capture.AllNSEMinutePredictor._fetch_yahoo_nifty_history(minute)

    assert len(result) == 2
    assert str(result.index.tz) == "Asia/Kolkata"
    assert result.index.max() == minute - pd.Timedelta(minutes=1)
    assert result["Close"].tolist() == [100.5, 101.5]




def test_model_hash_mismatch_refreshes_cached_artifacts_from_drive(monkeypatch, tmp_path):
    import hashlib
    import json
    import all_nse_intraday_model as model

    local_manifest = tmp_path / "manifest.json"
    local_classifier = tmp_path / "classifier.joblib"
    local_regressor = tmp_path / "regressor.joblib"
    monkeypatch.setattr(model, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(model, "LOCAL_MANIFEST", local_manifest)
    monkeypatch.setattr(model, "LOCAL_CLASSIFIER", local_classifier)
    monkeypatch.setattr(model, "LOCAL_REGRESSOR", local_regressor)

    classifier_bytes = b"verified-current-classifier"
    regressor_bytes = b"verified-current-regressor"
    manifest = {
        "status": "REPLACEMENT",
        "model_version": "unit-test",
        "classifier_sha256": hashlib.sha256(classifier_bytes).hexdigest(),
        "regressor_sha256": hashlib.sha256(regressor_bytes).hexdigest(),
        "training_symbols": ["AAA"],
        "training_symbol_count": 1,
    }
    local_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    local_classifier.write_bytes(b"stale-classifier")
    local_regressor.write_bytes(b"stale-regressor")

    downloads = []
    payloads = {
        model.REMOTE_MANIFEST: json.dumps(manifest).encode("utf-8"),
        model.REMOTE_CLASSIFIER: classifier_bytes,
        model.REMOTE_REGRESSOR: regressor_bytes,
    }

    def fake_drive_download(remote, local):
        downloads.append(remote)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(payloads[remote])

    monkeypatch.setattr(model, "_drive_download", fake_drive_download)

    status = model.model_status()

    assert status["manifest_sha256_match"] is True
    assert local_classifier.read_bytes() == classifier_bytes
    assert local_regressor.read_bytes() == regressor_bytes
    assert downloads == [
        model.REMOTE_MANIFEST,
        model.REMOTE_CLASSIFIER,
        model.REMOTE_REGRESSOR,
    ]

@pytest.mark.parametrize(
    "column_order",
    [
        [("Open", "^NSEI"), ("High", "^NSEI"), ("Low", "^NSEI"), ("Close", "^NSEI"), ("Volume", "^NSEI")],
        [("^NSEI", "Open"), ("^NSEI", "High"), ("^NSEI", "Low"), ("^NSEI", "Close"), ("^NSEI", "Volume")],
    ],
)
def test_yfinance_multindex_nifty_fallback_normalizes_real_ohlcv(
    monkeypatch, column_order
):
    import data_fetcher
    import yfinance as yf

    minute = pd.Timestamp("2026-10-27 10:00:00", tz=IST)
    idx = pd.date_range(
        minute - pd.Timedelta(minutes=2),
        periods=5,
        freq="min",
        tz=IST,
        name="Datetime",
    )
    values_by_field = {
        "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "High": [101.0, 102.0, 103.0, 104.0, 105.0],
        "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
        "Volume": [10.0, 12.0, 11.0, 13.0, 14.0],
    }
    frame = pd.DataFrame(
        {
            col: values_by_field[col[0] if col[0] != "^NSEI" else col[1]]
            for col in column_order
        },
        index=idx,
    )
    frame.columns = pd.MultiIndex.from_tuples(column_order, names=["Price", "Ticker"])
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    monkeypatch.setattr(data_fetcher, "_yahoo_chart_history", lambda *args, **kwargs: empty)
    monkeypatch.setattr(data_fetcher, "get_historical", lambda *args, **kwargs: empty)
    monkeypatch.setattr(
        yf,
        "Ticker",
        lambda *args, **kwargs: type("FakeTicker", (), {"history": lambda self, **kw: empty})(),
    )
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: frame)

    result = capture.AllNSEMinutePredictor._fetch_yahoo_nifty_history(minute)

    assert len(result) == 2
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert str(result.index.tz) == "Asia/Kolkata"
    assert result.index.max() == minute - pd.Timedelta(minutes=1)
    assert result["Close"].tolist() == [100.5, 101.5]
    assert result["Volume"].tolist() == [10.0, 12.0]


def test_yfinance_ticker_history_is_preferred_before_download(monkeypatch):
    import data_fetcher
    import yfinance as yf

    minute = pd.Timestamp("2026-10-27 10:00:00", tz=IST)
    idx = pd.date_range(
        minute - pd.Timedelta(minutes=2),
        periods=5,
        freq="min",
        tz=IST,
        name="Datetime",
    )
    ticker_frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [10.0, 12.0, 11.0, 13.0, 14.0],
        },
        index=idx,
    )
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    monkeypatch.setattr(data_fetcher, "_yahoo_chart_history", lambda *args, **kwargs: empty)
    monkeypatch.setattr(data_fetcher, "get_historical", lambda *args, **kwargs: empty)
    calls = []

    class FakeTicker:
        def history(self, **kwargs):
            calls.append(("ticker", kwargs))
            return ticker_frame

    monkeypatch.setattr(yf, "Ticker", lambda symbol: FakeTicker())

    def forbidden_download(*args, **kwargs):
        raise AssertionError("download fallback should not run when Ticker.history succeeds")

    monkeypatch.setattr(yf, "download", forbidden_download)

    result = capture.AllNSEMinutePredictor._fetch_yahoo_nifty_history(minute)

    assert len(result) == 2
    assert result["Close"].tolist() == [100.5, 101.5]
    assert str(result.index.tz) == "Asia/Kolkata"
    assert len(calls) == 1
    assert calls[0][1]["interval"] == "1m"

def test_unsupported_indstocks_index_id_is_not_used_as_rest_scrip_code():
    assert capture.AllNSEMinutePredictor._nifty_instrument_code() is None


def test_nifty_quote_uses_a_fresh_real_candle_when_no_rest_index_token_exists(tmp_path):
    predictor = capture.AllNSEMinutePredictor(tmp_path)
    now = pd.Timestamp.now(tz=IST)
    latest = now.floor("min") - pd.Timedelta(minutes=1)
    predictor._market_history = pd.DataFrame(
        {
            "Open": [24990.0],
            "High": [25010.0],
            "Low": [24980.0],
            "Close": [25000.0],
            "Volume": [1000.0],
        },
        index=pd.DatetimeIndex([latest], name="Datetime"),
    )

    price, observed_at, error = predictor._fetch_current_nifty_quote()

    assert price == pytest.approx(25000.0)
    assert observed_at == latest
    assert error is None


def test_session_history_refresh_is_nonblocking_and_waits_for_seed_completion(tmp_path, monkeypatch):
    import threading

    predictor = capture.AllNSEMinutePredictor(tmp_path)
    minute = pd.Timestamp("2026-10-27 10:20:00", tz=IST)
    calls = []
    completed = threading.Event()
    execution_context = []

    def fake_refresh(target):
        calls.append(target)
        execution_context.append(threading.current_thread() is not threading.main_thread())
        completed.set()
        return {
            "status": "OK",
            "minute": target.isoformat(),
            "batch_count": 1,
            "failed_batches": 0,
            "failed_symbols": [],
            "updated_symbols": 1,
            "current_session_candle_rows": 1,
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setattr(predictor, "_refresh_current_session_history", fake_refresh)

    # Bootstrap must not compete with its own full-universe history seeding.
    waiting = predictor._schedule_current_session_history_refresh(minute)
    assert waiting["status"] == "WARMING_SEED"
    assert calls == []

    predictor._seed_ready = True
    predictor._seed_complete = True
    predictor._codes = {"AAA": "NSE_100"}
    scheduled = predictor._schedule_current_session_history_refresh(minute)

    assert scheduled["status"] == "SCHEDULED_ASYNC"
    assert scheduled["batch_count"] == 0
    assert completed.wait(timeout=2.0)
    assert len(calls) == 1
    assert execution_context == [True]



def test_auth_recovery_errors_are_classified_without_masking_data_errors():
    assert capture._is_auth_recovery_error(
        "RuntimeError: INDstocks token generation failed: HTTP 400; provider_response=Please wait a min"
    )
    assert capture._is_auth_recovery_error(
        "INDstocks token generation is cooling down for another 55s"
    )
    assert not capture._is_auth_recovery_error(
        "historical refresh response did not report success"
    )
    assert not capture._is_auth_recovery_error(
        "No candles returned for inactive instrument"
    )


def test_history_refresh_stops_on_auth_failure_and_queues_unprocessed_symbols(
    tmp_path, monkeypatch
):
    import indstocks_client

    predictor = capture.AllNSEMinutePredictor(tmp_path)
    minute = pd.Timestamp("2026-10-27 10:20:00", tz=IST)
    symbols = [f"STOCK{i:02d}" for i in range(15)]
    predictor._seed_ready = True
    predictor._seed_complete = True
    predictor._codes = {symbol: f"NSE_{i}" for i, symbol in enumerate(symbols, start=1)}
    predictor._nifty_code = None
    monkeypatch.setattr(capture, "HISTORY_REFRESH_WORKERS", 1)
    monkeypatch.setattr(predictor, "_reserve_history_request_slot", lambda: None)
    monkeypatch.setattr(
        predictor,
        "_fetch_yahoo_nifty_history",
        lambda _minute: pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"],
            index=pd.DatetimeIndex([], tz=IST, name="Datetime"),
        ),
    )
    candle_ts = int((minute - pd.Timedelta(minutes=1)).timestamp())
    requests = []

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    def fake_request(method, path, *, params=None):
        requests.append((method, path, params))
        if len(requests) == 2:
            raise RuntimeError(
                "INDstocks token generation failed: HTTP 400; provider_response=Please wait a min"
            )
        codes = (params or {}).get("scrip-codes", "").split(",")
        return FakeResponse({
            "success": True,
            "data": {
                code: {"candles": [{
                    "ts": candle_ts, "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 10,
                }]}
                for code in codes
            },
        })

    monkeypatch.setattr(indstocks_client, "_request", fake_request)
    summary = predictor._refresh_current_session_history(minute)

    assert summary["status"] == "AUTH_DEFERRED"
    assert summary["auth_circuit_open"] is True
    assert summary["pending_symbol_count"] == 10
    assert summary["updated_symbols"] == 5
    assert summary["deferred_batches"] >= 1
    assert len(requests) == 2, "No further history batches should be sent after auth failure"
    assert predictor._history_refresh_pending_symbols == set(symbols[5:])
    assert summary["auth_error"]



def test_initial_history_seed_stops_after_shared_auth_failure(tmp_path, monkeypatch):
    import all_universe_minute_capture
    import delivery_training_universe
    import indstocks_client

    symbols = [f"STOCK{i}" for i in range(15)]
    codes = {symbol: f"NSE_{i}" for i, symbol in enumerate(symbols)}
    requests = []

    monkeypatch.setattr(all_universe_minute_capture, "_instrument_map", lambda: codes)
    monkeypatch.setattr(
        delivery_training_universe,
        "official_nse_symbols",
        lambda force_refresh=False: symbols,
    )
    monkeypatch.setattr(indstocks_client, "configured", lambda: True)

    def rejected_request(*args, **kwargs):
        requests.append((args, kwargs))
        raise RuntimeError(
            "INDstocks token generation failed: HTTP 400; provider_response=message"
        )

    monkeypatch.setattr(indstocks_client, "_request", rejected_request)
    predictor = capture.AllNSEMinutePredictor(tmp_path)
    monkeypatch.setattr(predictor, "_nifty_instrument_code", lambda: None)
    monkeypatch.setattr(
        predictor,
        "_fetch_yahoo_nifty_history",
        lambda minute: pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"],
            index=pd.DatetimeIndex([], tz=IST, name="Datetime"),
        ),
    )

    predictor._seed_history()

    status = predictor.status()
    assert len(requests) == 1, "a shared token failure must stop all remaining seed batches"
    assert status["history_seed_complete"] is False
    assert status["history_seed_failed_batches"] == 1
    assert status["history_seed_deferred_batches"] == 2
    assert status["status"] == "AUTH_DEFERRED"
    assert "token generation failed" in status["history_seed_error"]


def test_nifty_context_is_fetched_before_indstocks_instrument_auth(tmp_path, monkeypatch):
    import all_universe_minute_capture
    import indstocks_client

    def rejected_instrument_master():
        raise RuntimeError(
            "INDstocks token generation failed: HTTP 400; provider_response=message=Please wait a minute"
        )

    monkeypatch.setattr(
        all_universe_minute_capture, "_instrument_map", rejected_instrument_master
    )
    monkeypatch.setattr(indstocks_client, "configured", lambda: True)

    predictor = capture.AllNSEMinutePredictor(tmp_path)
    cutoff = pd.Timestamp.now(tz=IST).floor("min")
    idx = pd.date_range(
        end=cutoff - pd.Timedelta(minutes=1),
        periods=30,
        freq="min",
        tz=IST,
        name="Datetime",
    )
    nifty = pd.DataFrame(
        {
            "Open": np.linspace(24970, 25000, len(idx)),
            "High": np.linspace(24975, 25005, len(idx)),
            "Low": np.linspace(24965, 24995, len(idx)),
            "Close": np.linspace(24972, 25002, len(idx)),
            "Volume": np.arange(1, len(idx) + 1, dtype=float),
        },
        index=idx,
    )
    nifty.attrs["market_context_source"] = "test_yahoo"
    monkeypatch.setattr(predictor, "_fetch_yahoo_nifty_history", lambda minute: nifty)
    monkeypatch.setattr(
        capture.time,
        "sleep",
        lambda seconds: (_ for _ in ()).throw(AssertionError("auth failures must not sleep-loop")),
    )

    predictor._seed_history()

    assert len(predictor._market_history) == len(nifty)
    assert predictor._market_history["Close"].notna().all()
    status = predictor.status()
    assert status["status"] == "AUTH_DEFERRED"
    assert status["history_seed_complete"] is False
    assert "token generation failed" in status["history_seed_error"]

def test_history_seed_progress_is_published_before_universe_seed_finishes(tmp_path):
    predictor = capture.AllNSEMinutePredictor(tmp_path)
    predictor._symbols = ["AAA", "BBB", "CCC"]
    idx = pd.date_range(
        "2026-10-26 15:20:00",
        periods=3,
        freq="min",
        tz=IST,
        name="Datetime",
    )
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [10.0, 12.0, 11.0],
        },
        index=idx,
    )

    predictor._publish_history_seed_progress(
        {"__NIFTY50__": frame, "AAA": frame},
        coverage_symbols=1,
        failed_batches=0,
    )

    status = predictor.status()
    assert status["history_ready"] is True
    assert status["history_seed_complete"] is False
    assert status["history_seed_coverage_symbols"] == 1
    assert status["status"] == "READY_PARTIAL"
    assert "AAA" in predictor._history
    assert len(predictor._market_history) == 3

def test_history_seed_batches_never_mix_nifty_index_with_equities():
    entries = [(f"STOCK{i}", f"NSE_{i}") for i in range(1, 11)]

    batches = capture._history_seed_batches(entries, "NSE_99999", batch_size=5)

    assert [len(batch) for batch in batches] == [1, 5, 5]
    assert batches[0] == [("__NIFTY50__", "NSE_99999")]
    assert all(all(symbol != "__NIFTY50__" for symbol, _ in batch) for batch in batches[1:])
    assert [symbol for batch in batches[1:] for symbol, _ in batch] == [
        symbol for symbol, _ in entries
    ]


def test_history_seed_batches_work_without_nifty_index():
    entries = [(f"STOCK{i}", f"NSE_{i}") for i in range(1, 8)]

    batches = capture._history_seed_batches(entries, None, batch_size=5)

    assert [len(batch) for batch in batches] == [5, 2]
    assert [symbol for batch in batches for symbol, _ in batch] == [
        symbol for symbol, _ in entries
    ]

def test_nse_equity_universe_falls_back_to_indstocks_when_archive_is_blocked(tmp_path, monkeypatch):
    import requests
    import nse_data_layer

    monkeypatch.setattr(nse_data_layer, "NORMALIZED_ROOT", tmp_path / "normalized")
    monkeypatch.setattr(nse_data_layer, "RAW_ROOT", tmp_path / "raw")

    def blocked_archive(*_args, **_kwargs):
        raise requests.HTTPError("403 Forbidden")

    def fake_instruments(exchange):
        assert exchange == "NSE"
        return pd.DataFrame([
            {"TRADING_SYMBOL": "AAA", "SECURITY_ID": "100", "EXCH": "NSE", "SERIES": "EQ"},
            {"TRADING_SYMBOL": "BBB", "SECURITY_ID": "200", "EXCH": "NSE", "SERIES": "BE"},
            {"TRADING_SYMBOL": "BBB", "SECURITY_ID": "201", "EXCH": "NSE", "SERIES": "EQ"},
            {"TRADING_SYMBOL": "IGNORED", "SECURITY_ID": "300", "EXCH": "BSE", "SERIES": "EQ"},
            {"TRADING_SYMBOL": "T0TEST", "SECURITY_ID": "400", "EXCH": "NSE", "SERIES": "T0"},
        ])

    monkeypatch.setattr(nse_data_layer.requests, "get", blocked_archive)
    monkeypatch.setattr("indstocks_client._load_instruments", fake_instruments)

    first = nse_data_layer.fetch_nse_equity_universe(force_refresh=True)
    assert set(first["SYMBOL"]) == {"AAA", "BBB"}
    assert first["UNIVERSE_SOURCE"].eq("INDSTOCKS_NSE_INSTRUMENT_MASTER_FALLBACK").all()
    assert first.loc[first["SYMBOL"].eq("BBB"), "SECURITY_ID"].iloc[0] == "201"

    # A fresh fallback cache avoids repeatedly hitting the NSE archive during
    # every five-minute tick while that endpoint remains blocked.
    def should_not_call_archive(*_args, **_kwargs):
        raise AssertionError("cached universe should be used while fresh")

    monkeypatch.setattr(nse_data_layer.requests, "get", should_not_call_archive)
    second = nse_data_layer.fetch_nse_equity_universe(force_refresh=False)
    assert set(second["SYMBOL"]) == {"AAA", "BBB"}
    assert second["UNIVERSE_SOURCE"].eq("INDSTOCKS_NSE_INSTRUMENT_MASTER_FALLBACK").all()

