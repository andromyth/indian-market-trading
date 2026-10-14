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
