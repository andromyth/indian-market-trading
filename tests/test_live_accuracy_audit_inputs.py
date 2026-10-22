from __future__ import annotations

from datetime import date

import pandas as pd

from audit_autonomous_intraday_outcomes import evaluate_record
from audit_live_prediction_health import _expected_missing_sessions


def test_delivery_archive_audit_accepts_timestamp_serialized_master_date() -> None:
    available = {date(2026, 10, 5), date(2026, 10, 6)}
    missing, expected_count = _expected_missing_sessions(
        "2026-10-01 00:00:00", available
    )
    assert expected_count >= 2
    assert "2026-10-05" not in missing
    assert "2026-10-06" not in missing


def test_intraday_audit_marks_missing_decision_timestamp_invalid() -> None:
    bars = pd.DataFrame(
        {"close": [100.0]},
        index=pd.DatetimeIndex(["2026-10-15T09:15:00Z"]),
    )
    result = evaluate_record(
        {
            "symbol": "SBIN",
            "decision_timestamp": None,
            "horizon_min": 30,
            "signal": "BUY",
        },
        bars,
    )
    assert result["validation_status"] == "INVALID_PREDICTION"
    assert result["validation_reason"] == "missing_or_invalid_decision_timestamp"



def test_intraday_coverage_requires_populated_target_aligned_outcome(monkeypatch):
    import prediction_ledger
    import audit_live_prediction_health as audit

    monkeypatch.setattr(prediction_ledger, "load_records", lambda: [{
        "prediction_id": "intraday-1",
        "symbol": "SBIN",
        "mode": "Intraday",
        "prediction_horizon": "30m",
        "decision_timestamp": "2026-10-20T10:00:00+05:30",
    }])
    monkeypatch.setattr(prediction_ledger, "load_validation_records", lambda: [{
        "prediction_id": "intraday-1",
        "mode": "Intraday",
        "prediction_horizon": "30m",
        "validation_status": "WAITING",
        "actual_return_30m": None,
        "actual_direction_30m": None,
    }])

    result = audit._intraday_outcome_coverage()

    assert result["status"] == "NO_TARGET_ALIGNED_30M_OUTCOME_VALIDATOR"
    assert result["validation_rows_matched_by_prediction_id"] == 1
    assert result["matched_validated_rows"] == 0
    assert result["validation_rows_with_target_aligned_outcomes"] == 0
    assert result["accuracy_claim_allowed"] is False


def test_intraday_coverage_accepts_a_matched_validated_actual(monkeypatch):
    import prediction_ledger
    import audit_live_prediction_health as audit

    monkeypatch.setattr(prediction_ledger, "load_records", lambda: [{
        "prediction_id": "intraday-2",
        "symbol": "SBIN",
        "mode": "Intraday",
        "prediction_horizon": "30m",
        "decision_timestamp": "2026-10-20T10:00:00+05:30",
    }])
    monkeypatch.setattr(prediction_ledger, "load_validation_records", lambda: [{
        "prediction_id": "intraday-2",
        "mode": "Intraday",
        "prediction_horizon": "30m",
        "validation_status": "VALIDATED",
        "actual_return_30m": 0.01,
        "actual_direction_30m": "UP",
    }])

    result = audit._intraday_outcome_coverage()

    assert result["status"] == "TARGET_ALIGNED_OUTCOMES_AVAILABLE"
    assert result["validation_rows_with_target_aligned_outcomes"] == 1
    assert result["accuracy_claim_allowed"] is False



def test_drive_child_lookup_retries_transient_500_error(monkeypatch):
    import audit_autonomous_intraday_outcomes as audit

    class Response:
        status = 500

    class TransientDriveError(Exception):
        resp = Response()

    class Store:
        root_id = "root"

        def __init__(self):
            self.calls = 0

        def _find_child(self, parent, name, mime_type=None):
            self.calls += 1
            if self.calls == 1:
                raise TransientDriveError("Internal Error")
            return {"id": "folder-id", "name": name}

    store = Store()
    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setattr("random.random", lambda: 0.0)

    file_id = audit._resolve_drive_id(store, "root-folder/example.jsonl")

    assert file_id == "folder-id"
    # One transient retry, then a successful parent-folder lookup and file lookup.
    assert store.calls == 3


def test_drive_child_lookup_does_not_retry_confirmed_missing_file():
    import audit_autonomous_intraday_outcomes as audit

    class Store:
        root_id = "root"

        def __init__(self):
            self.calls = 0

        def _find_child(self, parent, name, mime_type=None):
            self.calls += 1
            return None

    store = Store()
    assert audit._resolve_drive_id(store, "missing/folder/file.parquet") is None
    assert store.calls == 1


def test_intraday_audit_retries_transient_drive_report_upload(tmp_path, monkeypatch):
    import audit_autonomous_intraday_outcomes as audit

    class Store:
        def __init__(self):
            self.calls = []

        def upload_or_update(self, local_path, remote_path, mime_type=None):
            self.calls.append((str(local_path), remote_path, mime_type))
            if len(self.calls) == 1:
                raise TimeoutError("The read operation timed out")
            return {"status": "OK", "remote_path": remote_path}

    store = Store()
    local = tmp_path / "outcomes.csv"
    local.write_text("prediction_id,actual_return\nid-1,0.01\n", encoding="utf-8")
    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setattr("random.random", lambda: 0.0)

    result = audit._upload_drive_with_retry(
        store,
        local,
        "production_data/intraday_predictions/outcomes_audit/outcomes.csv",
        mime_type="text/csv",
    )

    assert result["status"] == "OK"
    assert len(store.calls) == 2
    assert store.calls[0] == store.calls[1]


def test_intraday_coverage_ignores_other_horizons(monkeypatch):
    import prediction_ledger
    import audit_live_prediction_health as audit

    monkeypatch.setattr(prediction_ledger, "load_records", lambda: [
        {
            "prediction_id": "intraday-5m",
            "symbol": "SBIN",
            "mode": "Intraday",
            "prediction_horizon": "5m",
            "decision_timestamp": "2026-10-20T10:00:00+05:30",
        },
        {
            "prediction_id": "delivery-30m",
            "symbol": "SBIN",
            "mode": "Delivery",
            "prediction_horizon": "30m",
            "decision_timestamp": "2026-10-20T10:00:00+05:30",
        },
    ])
    monkeypatch.setattr(prediction_ledger, "load_validation_records", lambda: [])

    result = audit._intraday_outcome_coverage()

    assert result["status"] == "NO_30M_INTRADAY_PREDICTIONS"
    assert result["intraday_prediction_count"] == 0
    assert result["other_intraday_rows_excluded"] == 1
    assert result["accuracy_claim_allowed"] is False


def test_intraday_coverage_does_not_block_on_stale_targets_outside_lookback(monkeypatch):
    import prediction_ledger
    import audit_live_prediction_health as audit

    monkeypatch.setattr(prediction_ledger, "load_records", lambda: [{
        "prediction_id": "intraday-old-30m",
        "symbol": "SBIN",
        "mode": "Intraday",
        "prediction_horizon": "30m",
        "decision_timestamp": "2026-09-01T10:00:00+05:30",
    }])
    monkeypatch.setattr(prediction_ledger, "load_validation_records", lambda: [])

    result = audit._intraday_outcome_coverage()

    assert result["status"] == "NO_RECENT_MATURED_30M_PREDICTIONS"
    assert result["recent_matured_target_count"] == 0
    assert result["historical_outside_lookback_count"] == 1
    assert result["accuracy_claim_allowed"] is False


def test_intraday_coverage_does_not_block_on_future_targets(monkeypatch):
    import prediction_ledger
    import audit_live_prediction_health as audit

    monkeypatch.setattr(prediction_ledger, "load_records", lambda: [{
        "prediction_id": "intraday-future-30m",
        "symbol": "SBIN",
        "mode": "Intraday",
        "prediction_horizon": "30m",
        "decision_timestamp": (pd.Timestamp.now(tz="Asia/Kolkata") + pd.Timedelta(hours=1)).isoformat(),
        "signal_horizon_end": (pd.Timestamp.now(tz="Asia/Kolkata") + pd.Timedelta(hours=1, minutes=30)).isoformat(),
    }])
    monkeypatch.setattr(prediction_ledger, "load_validation_records", lambda: [])

    result = audit._intraday_outcome_coverage()

    assert result["status"] == "NO_RECENT_MATURED_30M_PREDICTIONS"
    assert result["recent_matured_target_count"] == 0
    assert result["future_target_count"] == 1
    assert result["accuracy_claim_allowed"] is False
