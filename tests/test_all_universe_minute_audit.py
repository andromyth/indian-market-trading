from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from audit_all_universe_minute_snapshots import (
    build_audit_report,
    build_prediction_audit,
    expected_minutes,
)


def _snapshot_frame(total=2, incomplete=0):
    rows = []
    complete_count = total - incomplete
    for index in range(total):
        complete = index < complete_count
        rows.append({
            "symbol": f"SYM{index:04d}",
            "instrument_code": f"NSE_{1000 + index}",
            "quote_status": "PASS" if complete else "MISSING_QUOTE",
            "actual_observed_price": 100.0 + index if complete else None,
            "quote_observed_at_utc": "2026-10-16T03:45:00+00:00",
        })
    return pd.DataFrame(rows)


def _complete_day_files(session_date):
    files = []
    manifests = {}
    snapshots = {}
    for key in expected_minutes(session_date):
        snapshot_id = f"snapshot-{key}"
        files.append({"id": snapshot_id, "name": f"snapshot_{key}.parquet"})
        snapshots[snapshot_id] = _snapshot_frame()
        manifest_id = f"manifest-{key}"
        files.append({"id": manifest_id, "name": f"manifest_{key}.json"})
        manifests[manifest_id] = {
            "status": "OK",
            "expected_symbol_count": 2,
            "quoted_symbol_count": 2,
            "missing_quote_count": 0,
            "unmapped_symbol_count": 0,
            "forecast_symbol_count": 2,
            "forecast_record_count": 4,
            "matched_outcome_count": 2,
            "pending_snapshot_recovery": {"errors": []},
        }
    return files, manifests, snapshots


def _prediction_snapshot_link(session_date, key):
    minute = datetime.strptime(key, "%Y%m%d_%H%M").replace(
        tzinfo=ZoneInfo("Asia/Kolkata")
    )
    return {
        "quote_snapshot_drive_path": (
            "production_data/all_universe_minute_snapshots/"
            f"{session_date:%Y/%m/%d}/snapshot_{key}.parquet"
        ),
        "quote_snapshot_cycle_minute_ist": minute.isoformat(),
        "quote_snapshot_capture_status": "OK",
        "quote_snapshot_expected_symbol_count": 2,
        "quote_snapshot_rows_validated": True,
    }


def _prediction_files(session_date):
    keys = expected_minutes(session_date)
    files = []
    manifests = {}
    for key in keys:
        manifest_id = f"prediction-manifest-{key}"
        files.extend([
            {"id": f"predictions-{key}", "name": f"predictions_{key}.parquet"},
            {"id": manifest_id, "name": f"manifest_{key}.json"},
        ])
        manifests[manifest_id] = {
            "status": "STORED",
            "expected_symbol_count": 2,
            "row_count": 1,
            "eligible_symbol_count": 1,
            "prediction_count": 1,
            "missing_quote_count": 1,
            "data_limited_count": 0,
            "untrained_symbol_count": 0,
            "model_failure_count": 0,
            "history_ready": False,
            "history_seed_coverage_symbols": 1,
            "history_seed_failed_batches": 0,
            "history_refresh_batch_count": 1,
            "history_refresh_failed_batches": 0,
            "model_artifact_verified": True,
            "trained_symbol_manifest_available": True,
            "prediction_status_counts": {"PREDICTED": 1},
            **_prediction_snapshot_link(session_date, key),
        }
    return keys, files, manifests


def test_expected_minutes_cover_open_through_last_five_minute_slot():
    keys = expected_minutes(date(2026, 10, 16))
    assert len(keys) == 75
    assert keys[0] == "20261016_0915"
    assert keys[-1] == "20261016_1525"
    assert all(
        (int(key[9:11]) * 60 + int(key[11:13]) - 555) % 5 == 0
        for key in keys
    )


def test_audit_records_all_stored_slots_without_a_universe_verdict():
    session_date = date(2026, 10, 16)
    files, manifests, snapshots = _complete_day_files(session_date)
    report, rows = build_audit_report(
        session_date, files, lambda file_id: manifests[file_id],
        snapshot_loader=lambda file_id: snapshots[file_id],
    )
    assert report["expected_minutes"] == 75
    assert report["snapshot_files_present"] == 75
    assert report["minute_manifests_present"] == 75
    assert report["observed_quote_coverage_pct"] == 100.0
    assert report["audit_status"] == "RECORDED"
    assert "quality_gate_pass" not in report
    assert "capture_quality_gate_pass" not in report
    assert len(rows) == 75


def test_one_incomplete_symbol_is_reported_but_does_not_fail_the_slot_or_day():
    session_date = date(2026, 10, 16)
    files, manifests, snapshots = _complete_day_files(session_date)
    partial_key = "20261016_1005"
    manifest = manifests[f"manifest-{partial_key}"]
    manifest.update({
        "status": "PARTIAL",
        "quoted_symbol_count": 1,
        "missing_quote_count": 1,
    })
    snapshots[f"snapshot-{partial_key}"] = _snapshot_frame(total=2, incomplete=1)

    report, rows = build_audit_report(
        session_date, files, lambda file_id: manifests[file_id],
        snapshot_loader=lambda file_id: snapshots[file_id],
    )
    row = rows.loc[rows["minute_key_ist"].eq(partial_key)].iloc[0]
    assert row["complete_symbol_count"] == 1
    assert row["incomplete_symbol_count"] == 1
    assert row["incomplete_symbol_samples"][0]["symbol"] == "SYM0001"
    assert partial_key in report["slots_with_incomplete_symbols"]
    assert report["audit_status"] == "RECORDED"
    assert "quality_gate_pass" not in report


def test_core_quote_fields_are_validated_per_symbol_without_a_coverage_threshold():
    session_date = date(2026, 10, 16)
    files, manifests, snapshots = _complete_day_files(session_date)
    key = "20261016_1005"
    frame = _snapshot_frame(total=10, incomplete=0)
    frame.loc[0, "instrument_code"] = None
    frame.loc[1, "actual_observed_price"] = None
    frame.loc[2, "quote_observed_at_utc"] = None
    snapshots[f"snapshot-{key}"] = frame
    manifests[f"manifest-{key}"].update({
        "expected_symbol_count": 10,
        "quoted_symbol_count": 10,
        "missing_quote_count": 0,
    })

    report, rows = build_audit_report(
        session_date, files, lambda file_id: manifests[file_id],
        snapshot_loader=lambda file_id: snapshots[file_id],
    )
    row = rows.loc[rows["minute_key_ist"].eq(key)].iloc[0]
    assert row["complete_symbol_count"] == 7
    assert row["incomplete_symbol_count"] == 3
    assert report["audit_status"] == "RECORDED"
    assert "quality_gate_pass" not in report


def test_recovered_persistence_error_is_recorded_without_percentage_gate():
    session_date = date(2026, 10, 16)
    files, manifests, snapshots = _complete_day_files(session_date)
    recovered_key = "20261016_1005"
    manifests[f"manifest-{recovered_key}"]["status"] = "PERSISTENCE_ERROR"

    report, _ = build_audit_report(
        session_date, files, lambda file_id: manifests[file_id],
        snapshot_loader=lambda file_id: snapshots[file_id],
    )
    assert recovered_key in report["recovered_persistence_issue_minutes"]
    assert report["audit_status"] == "RECORDED"


def test_expected_minutes_cap_live_session_at_last_completed_minute():
    ist = ZoneInfo("Asia/Kolkata")
    keys = expected_minutes(
        date(2026, 10, 16),
        as_of=datetime(2026, 10, 16, 9, 23, 30, tzinfo=ist),
    )
    assert len(keys) == 2
    assert keys[0] == "20261016_0915"
    assert keys[-1] == "20261016_0920"


def test_prediction_audit_accepts_partial_slot_with_individually_stored_predictions():
    session_date = date(2026, 10, 16)
    keys, files, manifests = _prediction_files(session_date)
    partial_key = "20261016_1010"
    manifests[f"prediction-manifest-{partial_key}"].update({
        "status": "PARTIAL",
        "expected_symbol_count": 2600,
        "row_count": 1,
        "eligible_symbol_count": 1,
        "prediction_count": 1,
        "missing_quote_count": 2599,
        "history_ready": False,
        "model_artifact_verified": False,
        "model_failure_count": 1,
    })
    manifests[f"prediction-manifest-{partial_key}"]["quote_snapshot_capture_status"] = "PARTIAL"

    report = build_prediction_audit(
        keys, files, lambda file_id: manifests[file_id], session_date=session_date
    )
    assert report["prediction_files_present"] == 75
    assert report["prediction_manifests_present"] == 75
    assert report["prediction_coverage_is_informational"] is True
    assert partial_key in {
        item["minute_key_ist"] for item in report["prediction_model_diagnostics"]
    }
    assert "prediction_quality_gate_pass" not in report
    assert report["total_model_prediction_count"] == 75


def test_prediction_audit_reports_missing_file_without_a_global_pass_fail():
    session_date = date(2026, 10, 16)
    keys, files, manifests = _prediction_files(session_date)
    missing_key = "20261016_1010"
    files = [item for item in files if item["name"] != f"predictions_{missing_key}.parquet"]
    report = build_prediction_audit(
        keys, files, lambda file_id: manifests[file_id], session_date=session_date
    )
    assert missing_key in report["missing_prediction_minutes"]
    assert "prediction_quality_gate_pass" not in report


def test_prediction_audit_reports_broken_snapshot_link_without_a_global_verdict():
    session_date = date(2026, 10, 16)
    keys, files, manifests = _prediction_files(session_date)
    broken_key = "20261016_0915"
    manifests[f"prediction-manifest-{broken_key}"]["quote_snapshot_drive_path"] = (
        "production_data/all_universe_minute_snapshots/2026/10/16/"
        "snapshot_20261016_0910.parquet"
    )
    report = build_prediction_audit(
        keys, files, lambda file_id: manifests[file_id], session_date=session_date
    )
    assert broken_key in report["prediction_snapshot_link_diagnostics"]
    assert "prediction_quality_gate_pass" not in report


def test_expected_minutes_before_open_is_empty_for_same_day():
    keys = expected_minutes(
        date(2026, 10, 16),
        as_of=datetime(2026, 10, 16, 8, 59, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    assert keys == []
