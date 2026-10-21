from __future__ import annotations

import pandas as pd

from prediction_ledger import _stable_id, prediction_revision_fingerprint
from delivery_prediction_csv_ledger import _latest_revision_per_symbol_date


def _record(**updates):
    record = {
        "symbol": "SBIN",
        "mode": "delivery",
        "prediction_horizon": "next_session",
        "decision_timestamp": "2026-10-15T15:30:00+05:30",
        "model_version": "model-v2",
        "prediction_for_date": "2026-10-16",
        "pit_snapshot_date": "2026-10-15",
        "signal": "HOLD",
        "raw_model_signal": "BUY",
        "signal_status": "DATA_INSUFFICIENT",
        "signal_reason": "EVENTS_STATE_UNKNOWN",
        "quality_gate_block_reasons": ["EVENTS_STATE_UNKNOWN"],
        "production_data_quality_gate": False,
        "probability_up_next_session": 0.57,
        "predicted_return_next_session": 0.012,
        "predicted_price_next_session": 101.2,
        "confidence": "LOW",
    }
    record.update(updates)
    return record


def test_same_prediction_output_is_idempotent_across_creation_times() -> None:
    first = _record(created_at="2026-10-16T03:00:00+00:00")
    second = _record(created_at="2026-10-16T04:00:00+00:00")

    assert prediction_revision_fingerprint(first) == prediction_revision_fingerprint(second)
    assert _stable_id(first) == _stable_id(second)


def test_changed_signal_or_quality_gate_creates_a_new_prediction_revision() -> None:
    original = _record()
    revised = _record(
        signal="BUY",
        signal_status="SCORED",
        signal_reason="",
        quality_gate_block_reasons=[],
        production_data_quality_gate=True,
    )

    assert prediction_revision_fingerprint(original) != prediction_revision_fingerprint(revised)
    assert _stable_id(original) != _stable_id(revised)


def test_changed_price_forecast_creates_a_new_prediction_revision() -> None:
    original = _record(predicted_price_next_session=101.2)
    revised = _record(predicted_price_next_session=103.4)

    assert _stable_id(original) != _stable_id(revised)


def test_daily_csv_view_selects_latest_revision_for_same_symbol_and_date() -> None:
    frame = pd.DataFrame([
        {
            "prediction_date": "2026-10-16",
            "symbol": "SBIN",
            "created_at": "2026-10-16T03:00:00+00:00",
            "prediction_id": "old-revision",
            "signal": "HOLD",
        },
        {
            "prediction_date": "2026-10-16",
            "symbol": "SBIN",
            "created_at": "2026-10-16T04:00:00+00:00",
            "prediction_id": "new-revision",
            "signal": "BUY",
        },
        {
            "prediction_date": "2026-10-16",
            "symbol": "RELIANCE",
            "created_at": "2026-10-16T03:30:00+00:00",
            "prediction_id": "rel-revision",
            "signal": "HOLD",
        },
    ])

    latest = _latest_revision_per_symbol_date(frame)

    assert len(latest) == 2
    sbin = latest.loc[latest["symbol"].eq("SBIN")].iloc[0]
    assert sbin["prediction_id"] == "new-revision"
    assert sbin["signal"] == "BUY"



def test_validation_loader_exists_and_reads_jsonl(tmp_path, monkeypatch):
    import json
    import prediction_ledger as ledger
    from pathlib import Path

    monkeypatch.setattr(ledger, "__file__", str(tmp_path / "prediction_ledger.py"))
    path = tmp_path / "data" / "prediction_ledger" / "prediction_validation.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"prediction_id": "p-1", "validation_status": "VALIDATED"},
        {"prediction_id": "p-2", "validation_status": "WAITING"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(ledger, "_drive_file_bytes", lambda _remote: None)

    assert ledger.load_validation_records() == rows


def test_save_validation_results_deduplicates_by_prediction_id(tmp_path, monkeypatch):
    import json
    import prediction_ledger as ledger

    monkeypatch.setattr(ledger, "__file__", str(tmp_path / "prediction_ledger.py"))
    monkeypatch.setattr(ledger, "_utc_timestamp", lambda: "2026-10-21T00:00:00+00:00")
    monkeypatch.setattr(ledger, "_drive_file_bytes", lambda _remote: None)
    rows = [
        {"prediction_id": "p-1", "validation_status": "WAITING"},
        {"prediction_id": "p-1", "validation_status": "VALIDATED", "actual_direction_d1": "UP"},
        {"prediction_id": "p-2", "validation_status": "VALIDATED", "actual_direction_d1": "DOWN"},
    ]

    result = ledger.save_validation_results(rows, sync_drive=False)
    assert result["status"] == "SAVED"
    assert result["rows"] == 2
    loaded = ledger.load_validation_records()
    latest = {row["prediction_id"]: row for row in loaded}
    assert latest["p-1"]["validation_status"] == "VALIDATED"
    assert latest["p-2"]["actual_direction_d1"] == "DOWN"



def test_append_predictions_is_idempotent_and_tracks_revisions(tmp_path, monkeypatch):
    import prediction_ledger as ledger

    local_dir = tmp_path / "data" / "prediction_ledger"
    monkeypatch.setattr(ledger, "LOCAL_DIR", local_dir)
    monkeypatch.setattr(ledger, "LOCAL_LEDGER", local_dir / "prediction_ledger.jsonl")
    monkeypatch.setattr(ledger, "_drive_file_bytes", lambda _remote: None)

    first = _record(created_at="2026-10-16T03:00:00+00:00")
    one = ledger.append_predictions([first], sync_drive=False)
    assert one["rows_added"] == 1
    assert one["synced"] is False

    repeated = _record(created_at="2026-10-16T04:00:00+00:00")
    same = ledger.append_predictions([repeated], sync_drive=False)
    assert same["rows_added"] == 0

    changed = _record(predicted_price_next_session=103.4)
    revised = ledger.append_predictions([changed], sync_drive=False)
    assert revised["rows_added"] == 1
    records = ledger.load_records()
    assert len(records) == 2
    assert len({row["prediction_id"] for row in records}) == 2
    assert sorted(row["prediction_revision"] for row in records) == [1, 2]


def test_explicit_path_keeps_minute_level_dataframe_loader(tmp_path):
    import json
    from prediction_ledger import load_records

    path = tmp_path / "minute.jsonl"
    path.write_text(json.dumps({
        "symbol": "SBIN",
        "decision_timestamp": "2026-10-21T09:30:00+05:30",
        "horizon_min": 5,
        "signal": "HOLD",
        "probability_up": 0.51,
        "reference_close": 100.0,
        "predicted_price": 100.1,
    }) + "\n", encoding="utf-8")
    loaded = load_records(path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["symbol"] == "SBIN"



def test_append_predictions_handles_legacy_fingerprint_in_revision_field(tmp_path, monkeypatch):
    import json
    import prediction_ledger as ledger

    local_dir = tmp_path / "data" / "prediction_ledger"
    local_dir.mkdir(parents=True)
    ledger_path = local_dir / "prediction_ledger.jsonl"
    monkeypatch.setattr(ledger, "LOCAL_DIR", local_dir)
    monkeypatch.setattr(ledger, "LOCAL_LEDGER", ledger_path)
    monkeypatch.setattr(ledger, "_drive_file_bytes", lambda _remote: None)

    legacy = _record(
        prediction_id="legacy-row",
        prediction_revision="37c7ccab7652d2d0",
        predicted_price_next_session=101.2,
    )
    ledger_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    changed = _record(predicted_price_next_session=103.4)
    result = ledger.append_predictions([changed], sync_drive=False)

    assert result["rows_added"] == 1
    assert result["synced"] is False
    rows = ledger.load_records()
    assert len(rows) == 2
    prior = next(row for row in rows if row["prediction_id"] == "legacy-row")
    newest = next(row for row in rows if row["prediction_id"] != "legacy-row")
    assert prior["prediction_revision"] == "37c7ccab7652d2d0"
    assert newest["prediction_revision"] == 2
