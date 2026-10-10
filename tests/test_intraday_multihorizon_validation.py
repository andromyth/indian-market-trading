from __future__ import annotations

import pandas as pd

import intraday_5m_multihorizon_validation as validator


def test_validate_one_uses_first_actual_bar_at_or_after_target(monkeypatch) -> None:
    monkeypatch.setattr(
        validator,
        "load_persistent_1m",
        lambda symbol: pd.DataFrame({
            "timestamp": ["2026-10-08T10:29:00+05:30", "2026-10-08T10:30:00+05:30", "2026-10-08T10:31:00+05:30"],
            "close": [99.0, 102.0, 103.0],
        }),
    )
    record = {
        "symbol": "RELIANCE.NS",
        "decision_timestamp": "2026-10-08T10:00:00+05:30",
        "horizon_minutes": 30,
        "current_price": 100.0,
        "predicted_price": 101.0,
        "probability_up": 0.7,
        "signal": "BUY",
    }
    result = validator.validate_one(record)
    assert result["validation_status"] == "VALIDATED"
    assert result["actual_price"] == 102.0
    assert result["target_alignment_seconds"] == 0.0
    assert result["direction_match"] is True
