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
