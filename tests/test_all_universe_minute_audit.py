from __future__ import annotations

from datetime import date

import pandas as pd

from audit_all_universe_minute_snapshots import build_audit_report, build_prediction_audit, expected_minutes


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
    for index, key in enumerate(expected_minutes(session_date)):
        snapshot_id = f"snapshot-{key}"
        files.append({
            "id": snapshot_id,
            "name": f"snapshot_{key}.parquet",
        })
        snapshots[snapshot_id] = _snapshot_frame()
        manifest_id = f"manifest-{key}"
        files.append({
            "id": manifest_id,
            "name": f"manifest_{key}.json",
        })
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


def test_expected_minutes_cover_open_through_last_five_minute_slot():
    keys = expected_minutes(date(2026, 10, 16))
    assert len(keys) == 75
    assert keys[0] == "20261016_0915"
    assert keys[-1] == "20261016_1525"
    assert all(
        (int(key[9:11]) * 60 + int(key[11:13]) - 555) % 5 == 0
        for key in keys
    )


def test_audit_passes_only_when_every_snapshot_and_manifest_is_present():
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
    assert report["quality_gate_pass"] is True
    assert report["audit_status"] == "PASS"
    assert len(rows) == 75


def test_audit_fails_when_a_minute_snapshot_is_missing():
    session_date = date(2026, 10, 16)
    files, manifests, snapshots = _complete_day_files(session_date)
    missing_key = "20261016_1005"
    files = [
        item for item in files
        if item["name"] != f"snapshot_{missing_key}.parquet"
    ]
    report, _ = build_audit_report(
        session_date, files, lambda file_id: manifests[file_id],
        snapshot_loader=lambda file_id: snapshots[file_id],
    )
    assert report["quality_gate_pass"] is False
    assert missing_key in report["missing_snapshot_minutes"]



def test_snapshot_capture_requires_more_than_90_percent_complete_symbols_in_every_slot():
    session_date = date(2026, 10, 16)
    files, manifests, snapshots = _complete_day_files(session_date)
    keys = expected_minutes(session_date)

    # Five slots each have only 1 of 2 complete symbols. Although whole-day
    # average quote coverage is high, these five individual snapshots must fail.
    for key in keys[:5]:
        manifest = manifests[f"manifest-{key}"]
        manifest.update({
            "status": "PARTIAL",
            "quoted_symbol_count": 1,
            "missing_quote_count": 1,
            "unmapped_symbol_count": 0,
        })
        snapshots[f"snapshot-{key}"] = _snapshot_frame(total=2, incomplete=1)

    report, _ = build_audit_report(
        session_date, files, lambda file_id: manifests[file_id],
        snapshot_loader=lambda file_id: snapshots[file_id],
    )
    assert report["observed_quote_coverage_pct"] > 90.0
    assert report["minimum_complete_symbol_coverage_pct"] == 50.0
    assert set(keys[:5]).issubset(report["failed_symbol_completeness_minutes"])
    assert report["capture_quality_gate_pass"] is False
    assert report["snapshot_quality_gate_pass"] is False
    assert report["quality_gate_pass"] is False
    assert report["audit_status"] == "NEEDS_ATTENTION"


def test_snapshot_capture_passes_when_each_slot_has_more_than_90_percent_complete_symbols():
    session_date = date(2026, 10, 16)
    files, manifests, snapshots = _complete_day_files(session_date)

    # 10 of 11 symbols have complete rows in every five-minute snapshot.
    for key in expected_minutes(session_date):
        manifest = manifests[f"manifest-{key}"]
        manifest.update({
            "status": "PARTIAL",
            "expected_symbol_count": 11,
            "quoted_symbol_count": 10,
            "missing_quote_count": 1,
            "unmapped_symbol_count": 0,
        })
        snapshots[f"snapshot-{key}"] = _snapshot_frame(total=11, incomplete=1)

    report, _ = build_audit_report(
        session_date, files, lambda file_id: manifests[file_id],
        snapshot_loader=lambda file_id: snapshots[file_id],
    )
    assert report["minimum_complete_symbol_coverage_pct"] > 90.0
    assert report["all_slots_meet_symbol_completeness_threshold"] is True
    assert report["capture_quality_gate_pass"] is True
    assert report["snapshot_quality_gate_pass"] is True
    assert report["quality_gate_pass"] is True
    assert report["audit_status"] == "PASS"


def test_snapshot_capture_does_not_pass_at_exactly_90_percent_complete_symbols():
    session_date = date(2026, 10, 16)
    files, manifests, snapshots = _complete_day_files(session_date)

    # Every slot has 9 of 10 symbols complete: exactly 90%, not greater than 90%.
    for key in expected_minutes(session_date):
        manifest = manifests[f"manifest-{key}"]
        manifest.update({
            "status": "PARTIAL",
            "expected_symbol_count": 10,
            "quoted_symbol_count": 9,
            "missing_quote_count": 1,
            "unmapped_symbol_count": 0,
        })
        snapshots[f"snapshot-{key}"] = _snapshot_frame(total=10, incomplete=1)

    report, _ = build_audit_report(
        session_date, files, lambda file_id: manifests[file_id],
        snapshot_loader=lambda file_id: snapshots[file_id],
    )
    assert report["observed_quote_coverage_pct"] == 90.0
    assert report["minimum_complete_symbol_coverage_pct"] == 90.0
    assert report["capture_quality_gate_pass"] is False
    assert report["quality_gate_pass"] is False
    assert report["audit_status"] == "NEEDS_ATTENTION"


def test_audit_accepts_recovered_drive_write_when_artifacts_are_present():
    session_date = date(2026, 10, 16)
    files, manifests, snapshots = _complete_day_files(session_date)
    recovered_key = "20261016_1005"
    recovered_id = f"manifest-{recovered_key}"
    manifests[recovered_id]["status"] = "PERSISTENCE_ERROR"

    report, _ = build_audit_report(
        session_date, files, lambda file_id: manifests[file_id],
        snapshot_loader=lambda file_id: snapshots[file_id],
    )
    assert report["quality_gate_pass"] is True
    assert recovered_key in report["recovered_persistence_issue_minutes"]


def test_expected_minutes_cap_live_session_at_last_completed_minute():
    from datetime import datetime

    from zoneinfo import ZoneInfo

    ist = ZoneInfo("Asia/Kolkata")
    keys = expected_minutes(
        date(2026, 10, 16),
        as_of=datetime(2026, 10, 16, 9, 23, 30, tzinfo=ist),
    )
    assert len(keys) == 2
    assert keys[0] == "20261016_0915"
    assert keys[-1] == "20261016_0920"


def test_prediction_audit_requires_all_slot_files_and_full_predictions_after_warmup():
    session_date = date(2026, 10, 16)
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
            "status": "OK",
            "expected_symbol_count": 2,
            "row_count": 2,
            "prediction_count": 2,
            "missing_quote_count": 0,
            "data_limited_count": 0,
            "untrained_symbol_count": 0,
            "model_failure_count": 0,
            "history_ready": True,
            "history_seed_coverage_symbols": 2,
            "history_seed_failed_batches": 0,
            "history_refresh_batch_count": 1,
            "history_refresh_failed_batches": 0,
            "model_artifact_verified": True,
            "trained_symbol_manifest_available": True,
            "quality_gate_pass": True,
        }
    report = build_prediction_audit(
        keys, files, lambda file_id: manifests[file_id], session_date=session_date
    )
    assert report["prediction_files_present"] == 75
    assert report["prediction_manifests_present"] == 75
    assert report["prediction_quality_gate_pass"] is True

    missing_key = "20261016_1010"
    files = [f for f in files if f["name"] != f"predictions_{missing_key}.parquet"]
    report = build_prediction_audit(
        keys, files, lambda file_id: manifests[file_id], session_date=session_date
    )
    assert report["prediction_quality_gate_pass"] is False
    assert missing_key in report["missing_prediction_minutes"]


def test_prediction_audit_fails_when_a_required_slot_is_partial():
    session_date = date(2026, 10, 16)
    keys = expected_minutes(session_date)
    files = []
    manifests = {}
    bad_key = "20261016_1010"
    for key in keys:
        manifest_id = f"prediction-manifest-{key}"
        files.extend([
            {"id": f"predictions-{key}", "name": f"predictions_{key}.parquet"},
            {"id": manifest_id, "name": f"manifest_{key}.json"},
        ])
        manifests[manifest_id] = {
            "status": "OK",
            "expected_symbol_count": 2,
            "row_count": 2,
            "prediction_count": 2,
            "missing_quote_count": 0,
            "data_limited_count": 0,
            "untrained_symbol_count": 0,
            "model_failure_count": 0,
            "history_ready": True,
            "history_seed_coverage_symbols": 2,
            "history_seed_failed_batches": 0,
            "history_refresh_batch_count": 1,
            "history_refresh_failed_batches": 0,
            "model_artifact_verified": True,
            "trained_symbol_manifest_available": True,
            "quality_gate_pass": True,
        }
    manifests[f"prediction-manifest-{bad_key}"]["status"] = "PARTIAL"
    manifests[f"prediction-manifest-{bad_key}"]["quality_gate_pass"] = False
    report = build_prediction_audit(
        keys, files, lambda file_id: manifests[file_id], session_date=session_date
    )
    assert report["prediction_quality_gate_pass"] is False
    assert bad_key in report["prediction_model_quality_failures"]


def test_expected_minutes_before_open_is_empty_for_same_day():
    from datetime import datetime

    from zoneinfo import ZoneInfo

    keys = expected_minutes(
        date(2026, 10, 16),
        as_of=datetime(2026, 10, 16, 8, 59, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    assert keys == []
