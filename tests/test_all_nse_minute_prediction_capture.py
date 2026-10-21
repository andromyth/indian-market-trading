from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import all_nse_minute_prediction_capture as capture

IST = ZoneInfo("Asia/Kolkata")


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
