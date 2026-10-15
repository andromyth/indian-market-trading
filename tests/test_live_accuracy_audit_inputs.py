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
