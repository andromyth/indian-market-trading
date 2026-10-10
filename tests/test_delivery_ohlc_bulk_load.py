from __future__ import annotations

import pandas as pd

import delivery_prediction_csv_ledger as ledger


def test_project_next_session_ohlc_uses_preloaded_eod_history(monkeypatch):
    history = pd.DataFrame(
        {
            "symbol": ["ABC"],
            "date": [pd.Timestamp("2026-10-08")],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
        }
    )

    def fail_if_archive_reloaded():
        raise AssertionError("Preloaded EOD history should prevent a second archive load")

    def capture_projection(symbol, reference_date, reference_close, predicted_close, eod):
        assert symbol == "ABC"
        assert reference_date == "2026-10-09"
        assert eod is history
        return {"status": "TEST_PASS"}

    monkeypatch.setattr(ledger, "_load_eod", fail_if_archive_reloaded)
    monkeypatch.setattr(ledger, "_project_ohlc", capture_projection)

    result = ledger.project_next_session_ohlc(
        "ABC", "2026-10-09", 101.0, 102.0, eod_history=history
    )

    assert result["status"] == "TEST_PASS"
