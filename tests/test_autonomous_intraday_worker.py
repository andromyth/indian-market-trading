from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from autonomous_intraday_worker import (
    current_session_completed,
    is_fresh_current_session,
    normalize_bars,
    segment_should_stop,
)
from selfdriving.sd_intraday_v2 import _session_rows

IST = ZoneInfo("Asia/Kolkata")


def _bars(start: str, count: int, *, tz=IST) -> pd.DataFrame:
    index = pd.date_range(start, periods=count, freq="min", tz=tz)
    close = 100 + np.arange(count) * 0.03 + np.sin(np.arange(count) / 4) * 0.1
    return pd.DataFrame(
        {
            "Open": close - 0.01,
            "High": close + 0.05,
            "Low": close - 0.05,
            "Close": close,
            "Volume": np.full(count, 1000.0),
        },
        index=index,
    )


def test_segment_worker_stops_at_expected_nse_market_bounds():
    assert segment_should_stop(datetime(2026, 10, 13, 12, 15, tzinfo=IST), "morning")
    assert not segment_should_stop(datetime(2026, 10, 13, 12, 14, tzinfo=IST), "morning")
    assert not segment_should_stop(datetime(2026, 10, 13, 15, 29, tzinfo=IST), "afternoon")
    assert segment_should_stop(datetime(2026, 10, 13, 15, 30, tzinfo=IST), "afternoon")
    assert segment_should_stop(datetime(2026, 10, 17, 10, 0, tzinfo=IST), "once")


def test_normalize_and_freshness_use_current_completed_candles():
    now = pd.Timestamp("2026-10-13 11:00:30", tz=IST)
    raw = _bars("2026-10-13 10:55", 6)
    raw.index.name = "timestamp"
    normalized = normalize_bars(raw)
    completed = current_session_completed(normalized, now)
    fresh, details = is_fresh_current_session(completed, now)

    assert len(completed) == 5
    assert fresh is True
    assert details["status"] == "PASS"
    assert completed.index.max() == pd.Timestamp("2026-10-13 10:59:00", tz=IST)


def test_normalize_bars_handles_duplicate_yahoo_multilevel_fields():
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "HCLTECH.NS"),
            ("High", "HCLTECH.NS"),
            ("Low", "HCLTECH.NS"),
            ("Close", "HCLTECH.NS"),
            ("Volume", "HCLTECH.NS"),
            ("Open", "duplicate-provider-column"),
        ]
    )
    raw = pd.DataFrame(
        [[100.0, 101.0, 99.0, 100.5, 1200.0, 999.0]],
        columns=columns,
        index=pd.DatetimeIndex(["2026-10-13 10:00:00"], tz=IST),
    )

    normalized = normalize_bars(raw)

    assert not normalized.empty
    assert normalized.iloc[0]["Open"] == 100.0
    assert normalized.iloc[0]["High"] == 101.0
    assert normalized.iloc[0]["Close"] == 100.5
    assert normalized.columns.is_unique


def test_worker_rejects_stale_candles():
    now = pd.Timestamp("2026-10-13 11:10:00", tz=IST)
    old = _bars("2026-10-13 10:00", 5)
    fresh, details = is_fresh_current_session(old, now)
    assert fresh is False
    assert details["status"] == "STALE"


def test_live_inference_can_emit_minute_anchors_while_training_default_stays_five_minutes():
    prices = _bars("2026-10-13 09:15", 100)
    prices["symbol"] = "HCLTECH"
    market = _bars("2026-10-13 09:15", 100)

    minute_rows = _session_rows(
        prices,
        market,
        (5, 15, 30),
        include_future=False,
        decision_interval_minutes=1,
    )
    default_rows = _session_rows(
        prices,
        market,
        (5, 15, 30),
        include_future=False,
    )

    minute_anchors = {
        pd.Timestamp(row["decision_timestamp"])
        for row in minute_rows
    }
    default_anchors = {
        pd.Timestamp(row["decision_timestamp"])
        for row in default_rows
    }

    assert len(minute_anchors) > len(default_anchors) * 4
    assert len(default_anchors) >= 10
    assert all(ts.second == 0 for ts in minute_anchors)


def test_cloud_run_worker_tick_skips_weekend_without_fetching_or_persisting(tmp_path):
    from autonomous_intraday_worker import AutonomousIntradayWorker

    worker = AutonomousIntradayWorker(
        segment="service",
        root=tmp_path,
        flush_every=5,
    )
    result = worker.tick_once(datetime(2026, 10, 17, 10, 15, tzinfo=IST))

    assert result["status"] == "MARKET_CLOSED"
    assert worker.cycle_count == 0
    assert worker.run_status == "STARTING"


def test_cloud_run_worker_runs_at_most_one_cycle_per_minute(tmp_path, monkeypatch):
    from autonomous_intraday_worker import AutonomousIntradayWorker

    worker = AutonomousIntradayWorker(
        segment="service",
        root=tmp_path,
        flush_every=5,
    )
    monkeypatch.setattr(worker, "_hydrate_daily_ledger", lambda: None)
    calls = []
    monkeypatch.setattr(
        worker,
        "_cycle",
        lambda: calls.append("cycle") or {"errors": [], "predictions_written": 2},
    )
    tick_time = datetime(2026, 10, 13, 11, 15, 30, tzinfo=IST)

    first = worker.tick_once(tick_time)
    duplicate = worker.tick_once(tick_time.replace(second=50))

    assert first["status"] == "PASS"
    assert duplicate["status"] == "DUPLICATE_MINUTE_SKIPPED"
    assert len(calls) == 1
    assert worker.cycle_count == 1



def test_beta_v2_runtime_imports_numpy_alias():
    from selfdriving import sd_v2_runtime

    assert sd_v2_runtime.np is np


def test_prediction_health_distinguishes_runtime_failure_from_valid_abstain():
    from autonomous_intraday_worker import _prediction_row_failure

    failure = _prediction_row_failure(
        "RBLBANK",
        {
            "horizon_min": 5,
            "signal": "ABSTAIN",
            "reason": "NameError: name 'np' is not defined",
        },
    )
    assert failure == {
        "symbol": "RBLBANK",
        "horizon_min": 5,
        "error": "Prediction unavailable: NameError: name 'np' is not defined",
    }

    valid_abstain = _prediction_row_failure(
        "RBLBANK",
        {
            "horizon_min": 5,
            "signal": "ABSTAIN",
            "signal_reason": "NO_VALIDATED_ACTION_BAND",
            "probability_up": 0.51,
            "decision_timestamp": "2026-10-14T05:09:00Z",
        },
    )
    assert valid_abstain is None


def test_cloud_run_partial_prediction_tick_flushes_immediately(tmp_path, monkeypatch):
    from autonomous_intraday_worker import AutonomousIntradayWorker

    worker = AutonomousIntradayWorker(
        segment="service",
        root=tmp_path,
        flush_every=5,
    )
    monkeypatch.setattr(worker, "_hydrate_daily_ledger", lambda: None)
    monkeypatch.setattr(
        worker,
        "_cycle",
        lambda: {
            "errors": [
                {"symbol": "RBLBANK", "horizon_min": 5, "error": "Prediction unavailable"}
            ],
            "predictions_written": 6,
            "prediction_rows_valid": 0,
            "prediction_errors": 6,
            "prediction_abstains": 0,
        },
    )
    flush_calls = []
    monkeypatch.setattr(
        worker,
        "_flush",
        lambda final=False: (
            flush_calls.append(final)
            or {
                "status": "OK",
                "data_partitions_synced": 5,
                "ledger_synced": True,
                "status_synced": True,
                "errors": [],
            }
        ),
    )

    result = worker.tick_once(datetime(2026, 10, 14, 10, 15, 30, tzinfo=IST))

    assert result["status"] == "PARTIAL"
    assert result["flush"]["status"] == "OK"
    assert flush_calls == [False]



def test_worker_health_records_empty_prediction_output_as_failure(tmp_path, monkeypatch):
    from autonomous_intraday_worker import AutonomousIntradayWorker

    worker = AutonomousIntradayWorker(
        segment="service",
        root=tmp_path,
        flush_every=5,
    )
    monkeypatch.setattr(worker, "_hydrate_daily_ledger", lambda: None)
    monkeypatch.setattr(
        worker,
        "_cycle",
        lambda: {
            "errors": [
                {
                    "symbol": "RBLBANK",
                    "error": "Intraday runtime produced no prediction rows after warmup",
                }
            ],
            "predictions_written": 0,
            "prediction_rows_valid": 0,
            "prediction_errors": 1,
            "prediction_abstains": 0,
            "warmup_symbols": [
                {"symbol": "RBLBANK", "rows": 85, "required": 80,
                 "reason": "runtime produced no eligible rows"}
            ],
        },
    )
    flush_calls = []
    monkeypatch.setattr(
        worker,
        "_flush",
        lambda final=False: (
            flush_calls.append(final)
            or {
                "status": "OK",
                "data_partitions_synced": 5,
                "ledger_synced": True,
                "status_synced": True,
                "errors": [],
            }
        ),
    )

    result = worker.tick_once(datetime(2026, 10, 14, 10, 16, 30, tzinfo=IST))

    assert result["status"] == "PARTIAL"
    assert result["summary"]["prediction_errors"] == 1
    assert result["flush"]["status"] == "OK"
    assert flush_calls == [False]


def test_worker_health_endpoint_retains_specific_prediction_error(monkeypatch):
    from autonomous_intraday_worker_service import STATE, _record_tick_result

    original = dict(STATE)
    try:
        _record_tick_result({
            "status": "PARTIAL",
            "summary": {
                "errors": [
                    {
                        "symbol": "RBLBANK",
                        "horizon_min": 5,
                        "error": "Prediction unavailable: NameError: name 'np' is not defined",
                    }
                ]
            },
        })
        assert "NameError: name 'np' is not defined" in STATE["last_error"]
    finally:
        STATE.clear()
        STATE.update(original)


def test_cloud_run_worker_final_minute_flushes_successful_prediction_batch(tmp_path, monkeypatch):
    from autonomous_intraday_worker import AutonomousIntradayWorker

    worker = AutonomousIntradayWorker(
        segment="service",
        root=tmp_path,
        flush_every=5,
    )
    monkeypatch.setattr(worker, "_hydrate_daily_ledger", lambda: None)
    monkeypatch.setattr(
        worker,
        "_cycle",
        lambda: {
            "errors": [],
            "predictions_written": 30,
            "prediction_rows_valid": 30,
            "prediction_errors": 0,
            "prediction_abstains": 0,
        },
    )
    flush_calls = []
    monkeypatch.setattr(
        worker,
        "_flush",
        lambda final=False: (
            flush_calls.append(final)
            or {
                "status": "OK",
                "data_partitions_synced": 5,
                "ledger_synced": True,
                "status_synced": True,
                "errors": [],
            }
        ),
    )

    result = worker.tick_once(datetime(2026, 10, 13, 15, 29, 30, tzinfo=IST))

    assert result["status"] == "PASS"
    assert result["flush"]["status"] == "OK"
    assert flush_calls == [True]
    assert result["worker_status"] == "STOPPED"


def test_cloud_run_closed_tick_retries_final_flush_when_local_ledger_exists(tmp_path, monkeypatch):
    from autonomous_intraday_worker import AutonomousIntradayWorker

    worker = AutonomousIntradayWorker(
        segment="service",
        root=tmp_path,
        flush_every=5,
    )
    closed_time = datetime(2026, 10, 17, 10, 15, tzinfo=IST)
    worker._ensure_paths(pd.Timestamp(closed_time))
    assert worker.daily_ledger is not None
    worker.daily_ledger.write_text('{"prediction_id":"unsynced-test-row"}\\n', encoding="utf-8")

    flush_calls = []
    monkeypatch.setattr(
        worker,
        "_flush",
        lambda final=False: (
            flush_calls.append(final)
            or {
                "status": "OK",
                "data_partitions_synced": 0,
                "ledger_synced": True,
                "status_synced": True,
                "errors": [],
            }
        ),
    )

    result = worker.tick_once(closed_time)

    assert result["status"] == "MARKET_CLOSED"
    assert result["run_status"] == "STOPPED"
    assert flush_calls == [True]



def test_minute_snapshot_persists_prediction_and_actual_price_to_drive(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from autonomous_intraday_worker import AutonomousIntradayWorker

    uploads = []

    class FakeDriveStore:
        def upload_or_update(self, local_path, remote_path, mime_type=None):
            with open(local_path, encoding="utf-8") as handle:
                content = handle.read()
            uploads.append({
                "remote_path": remote_path,
                "mime_type": mime_type,
                "content": content,
            })
            return {"status": "updated", "path": remote_path}

    monkeypatch.setitem(
        sys.modules, "persistent_store", SimpleNamespace(GoogleDriveStore=FakeDriveStore)
    )
    worker = AutonomousIntradayWorker(segment="service", root=tmp_path)
    worker.last_minute_snapshot_rows = [{
        "record_type": "intraday_minute_observation",
        "symbol": "SBIN",
        "actual_observed_close": 812.35,
        "actual_observed_timestamp_ist": "2026-10-14T10:14:00+05:30",
        "predictions": [{
            "horizon_min": 5,
            "signal": "BUY",
            "predicted_price": 814.0,
            "outcome_status": "WAITING_FOR_HORIZON",
        }],
    }]

    result = worker._persist_minute_snapshot(
        pd.Timestamp("2026-10-14 10:15:00", tz=IST), {"status": "PASS"}
    )

    assert result["status"] == "OK"
    assert result["snapshot_rows"] == 1
    assert result["prediction_rows"] == 1
    assert uploads[0]["remote_path"] == (
        "production_data/intraday_minute_snapshots/2026/10/14/"
        "snapshot_20261014_1015.jsonl"
    )
    saved = json.loads(uploads[0]["content"].strip())
    assert saved["actual_observed_close"] == 812.35
    assert saved["predictions"][0]["predicted_price"] == 814.0
    assert saved["predictions"][0]["outcome_status"] == "WAITING_FOR_HORIZON"



def test_service_drive_preflight_failure_is_diagnostic_not_startup_exception(monkeypatch):
    import persistent_store
    from autonomous_intraday_worker_service import STATE, _preflight

    original = dict(STATE)
    monkeypatch.setattr(
        persistent_store,
        "drive_status",
        lambda require_write=True: {
            "configured": True,
            "reachable": False,
            "write_ready": False,
            "auth_method": "oauth",
            "reason": "simulated expired OAuth token",
        },
    )
    try:
        result = _preflight()
        assert result["ready"] is False
        assert STATE["drive_preflight_status"] == "ERROR"
        assert "simulated expired OAuth token" in STATE["drive_preflight"]["reason"]
        assert "Google Drive preflight failed" in STATE["last_error"]
    finally:
        STATE.clear()
        STATE.update(original)
