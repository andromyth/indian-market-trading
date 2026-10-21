from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import all_universe_minute_capture as capture

IST = ZoneInfo("Asia/Kolkata")


def test_outcome_match_uses_actual_quote_in_target_minute_only():
    target = pd.Timestamp("2026-10-16 10:05:00", tz=IST)
    observed = pd.Timestamp("2026-10-16 10:05:24", tz=IST)
    forecasts = [
        {
            "forecast_key": "AAA|target-1",
            "symbol": "AAA",
            "target_timestamp_utc": target.tz_convert("UTC").isoformat(),
            "horizon_min": 5,
            "reference_close": 100.0,
            "predicted_price": 103.0,
            "signal": "BUY",
        },
        {
            "forecast_key": "AAA|target-2",
            "symbol": "AAA",
            "target_timestamp_utc": (target + pd.Timedelta(minutes=1)).tz_convert("UTC").isoformat(),
            "horizon_min": 5,
            "reference_close": 100.0,
            "predicted_price": 103.0,
            "signal": "BUY",
        },
    ]

    outcomes = capture._outcomes_for_observation("AAA", 101.0, observed, forecasts)

    assert len(outcomes) == 1
    assert outcomes[0]["forecast_key"] == "AAA|target-1"
    assert outcomes[0]["actual_target_price"] == 101.0
    assert outcomes[0]["actual_return_pct_from_reference"] == pytest.approx(1.0)
    assert outcomes[0]["absolute_price_error"] == 2.0
    assert outcomes[0]["directional_correct_before_costs"] is True
    assert outcomes[0]["outcome_status"] == "MATCHED_TARGET_MINUTE_QUOTE"


def test_quote_row_does_not_claim_unavailable_stock_was_forecast():
    now = pd.Timestamp("2026-10-16 10:05:24", tz=IST)
    row = capture._quote_row(
        "BBB",
        "NSE_2",
        {"live_price": 50.2, "day_open": 49.5, "volume": 1000},
        now,
        now.floor("min"),
        {},
        [],
    )

    assert row["quote_status"] == "PASS"
    assert row["actual_observed_price"] == 50.2
    assert row["forecast_status"] == "NO_FORECAST_EMITTED_FOR_SYMBOL"
    assert row["is_one_minute_ohlcv_candle"] is False
    assert json.loads(row["current_forecasts_json"]) == []
    assert json.loads(row["matched_outcomes_json"]) == []


def test_full_universe_snapshot_persists_quote_and_missing_symbol_rows(
    tmp_path, monkeypatch
):
    master = pd.DataFrame(
        [
            {"EXCH": "NSE", "SECURITY_ID": "1", "TRADING_SYMBOL": "AAA", "SERIES": "EQ"},
            {"EXCH": "NSE", "SECURITY_ID": "2", "TRADING_SYMBOL": "BBB", "SERIES": "EQ"},
        ]
    )

    class FakeResponse:
        def json(self):
            return {
                "status": "success",
                "data": {
                    "NSE_1": {
                        "live_price": 100.5,
                        "day_open": 99.5,
                        "day_high": 101.0,
                        "day_low": 98.9,
                        "prev_close": 99.0,
                        "volume": 12345,
                        "day_change_percentage": 1.52,
                    }
                },
            }

    class FakeDriveStore:
        def __init__(self):
            self.uploads = []

        def upload_or_update(self, local_path, remote_path, mime_type=None):
            local = Path(local_path)
            self.uploads.append(
                {
                    "remote_path": remote_path,
                    "mime_type": mime_type,
                    "bytes": local.read_bytes(),
                }
            )
            return {"status": "OK", "remote_path": remote_path, "size": local.stat().st_size}

    fake_store = FakeDriveStore()
    fake_indstocks = SimpleNamespace(
        _load_instruments=lambda exchange: master,
        configured=lambda: True,
        _request=lambda method, path, params=None: FakeResponse(),
    )
    monkeypatch.setitem(__import__("sys").modules, "indstocks_client", fake_indstocks)
    monkeypatch.setattr(
        "delivery_training_universe.official_nse_symbols",
        lambda force_refresh=False: ["AAA", "BBB"],
    )
    monkeypatch.setattr("persistent_store.GoogleDriveStore", lambda: fake_store)

    result = capture.capture_all_universe_minute_snapshot(
        now=pd.Timestamp("2026-10-16 10:05:12", tz=IST),
        worker_root=tmp_path,
        daily_ledger=None,
        prediction_snapshots=[],
    )

    assert result["status"] == "PARTIAL"
    assert result["expected_symbol_count"] == 2
    assert result["quoted_symbol_count"] == 1
    assert result["missing_quote_count"] == 1
    assert result["row_count"] == 2
    remote_snapshot = (
        "production_data/all_universe_minute_snapshots/2026/10/16/"
        "snapshot_20261016_1005.parquet"
    )
    uploaded = next(item for item in fake_store.uploads if item["remote_path"] == remote_snapshot)
    assert any(
        item["remote_path"]
        == "production_data/all_universe_minute_snapshots/2026/10/16/manifest_20261016_1005.json"
        for item in fake_store.uploads
    )
    assert any(
        item["remote_path"]
        == "production_data/all_universe_minute_snapshots/latest_manifest.json"
        for item in fake_store.uploads
    )
    local_parquet = tmp_path / "copy.parquet"
    local_parquet.write_bytes(uploaded["bytes"])
    frame = pd.read_parquet(local_parquet).set_index("symbol")

    assert frame.loc["AAA", "actual_observed_price"] == 100.5
    assert frame.loc["AAA", "quote_status"] == "PASS"
    assert frame.loc["BBB", "quote_status"] == "MISSING_QUOTE"
    assert frame.loc["BBB", "actual_observed_price"] != frame.loc["BBB", "actual_observed_price"]



def test_pending_snapshot_upload_is_retried_once_and_receipt_prevents_reupload(tmp_path):
    class FakeDriveStore:
        def __init__(self):
            self.paths = []

        def upload_or_update(self, local_path, remote_path, mime_type=None):
            self.paths.append(remote_path)
            return {"status": "OK", "remote_path": remote_path}

    local_dir = tmp_path / "all_universe_minute_snapshots"
    local_dir.mkdir()
    pending = local_dir / "snapshot_20261016_1005.parquet"
    pending.write_bytes(b"prior-local-snapshot")
    store = FakeDriveStore()

    recovered = capture._upload_pending_snapshots(
        store, local_dir, "snapshot_20261016_1006.parquet"
    )

    assert recovered["attempted"] == 1
    assert recovered["recovered"] == 1
    assert recovered["errors"] == []
    assert store.paths == [
        "production_data/all_universe_minute_snapshots/2026/10/16/"
        "snapshot_20261016_1005.parquet"
    ]
    assert pending.with_suffix(".uploaded.json").is_file()

    second_attempt = capture._upload_pending_snapshots(
        store, local_dir, "snapshot_20261016_1006.parquet"
    )
    assert second_attempt["attempted"] == 0
    assert second_attempt["recovered"] == 0
    assert len(store.paths) == 1


def test_service_health_clears_recovered_quote_coverage_warning():
    from autonomous_intraday_worker_service import STATE, _record_tick_result

    original = dict(STATE)
    try:
        STATE["all_universe_capture_error"] = None
        STATE["all_universe_capture_warning"] = None
        STATE["last_error"] = None
        _record_tick_result({
            "status": "PASS",
            "all_universe_capture": {
                "status": "PARTIAL",
                "expected_symbol_count": 2600,
                "quoted_symbol_count": 2598,
                "missing_quote_count": 2,
            },
        })
        assert STATE["all_universe_capture_warning"] == (
            "Quote coverage incomplete: 2598/2600 NSE symbols"
        )
        assert STATE["last_error"] == STATE["all_universe_capture_warning"]

        _record_tick_result({
            "status": "PASS",
            "all_universe_capture": {
                "status": "OK",
                "expected_symbol_count": 2600,
                "quoted_symbol_count": 2600,
                "missing_quote_count": 0,
            },
        })
        assert STATE["all_universe_capture_warning"] is None
        assert STATE["all_universe_capture_error"] is None
        assert STATE["last_error"] is None
    finally:
        STATE.clear()
        STATE.update(original)



def test_current_forecast_map_excludes_stale_snapshot_rows():
    cycle_minute = pd.Timestamp("2026-10-16 10:05:00", tz=IST)
    snapshot_rows = [{
        "symbol": "AAA",
        "predictions": [
            {
                "forecast_key": "new",
                "worker_observed_at_utc": "2026-10-16T04:35:12+00:00",
            },
            {
                "forecast_key": "stale",
                "worker_observed_at_utc": "2026-10-16T04:34:12+00:00",
            },
        ],
    }]
    result = capture._prediction_map(snapshot_rows, cycle_minute)
    assert [row["forecast_key"] for row in result["AAA"]] == ["new"]


def test_instrument_map_keeps_supported_non_eq_series_and_prefers_eq(monkeypatch):
    master = pd.DataFrame([
        {"EXCH": "NSE", "SECURITY_ID": "11", "TRADING_SYMBOL": "AAA", "SERIES": "BE"},
        {"EXCH": "NSE", "SECURITY_ID": "22", "TRADING_SYMBOL": "BBB", "SERIES": "BE"},
        {"EXCH": "NSE", "SECURITY_ID": "23", "TRADING_SYMBOL": "BBB", "SERIES": "EQ"},
        {"EXCH": "NSE", "SECURITY_ID": "33", "TRADING_SYMBOL": "CCC", "SERIES": "BZ"},
        {"EXCH": "NSE", "SECURITY_ID": "44", "TRADING_SYMBOL": "DDD", "SERIES": "IV"},
    ])
    fake_indstocks = SimpleNamespace(_load_instruments=lambda exchange: master)
    monkeypatch.setitem(__import__("sys").modules, "indstocks_client", fake_indstocks)

    mapped = capture._instrument_map()

    assert mapped == {"AAA": "NSE_11", "BBB": "NSE_23", "CCC": "NSE_33"}
