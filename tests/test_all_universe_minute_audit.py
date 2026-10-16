from __future__ import annotations

from datetime import date

from audit_all_universe_minute_snapshots import build_audit_report, expected_minutes


def _complete_day_files(session_date):
    files = []
    manifests = {}
    for index, key in enumerate(expected_minutes(session_date)):
        files.append({
            "id": f"snapshot-{key}",
            "name": f"snapshot_{key}.parquet",
        })
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
    return files, manifests


def test_expected_minutes_cover_open_through_last_regular_session_minute():
    keys = expected_minutes(date(2026, 10, 16))
    assert len(keys) == 375
    assert keys[0] == "20261016_0915"
    assert keys[-1] == "20261016_1529"


def test_audit_passes_only_when_every_snapshot_and_manifest_is_present():
    session_date = date(2026, 10, 16)
    files, manifests = _complete_day_files(session_date)
    report, rows = build_audit_report(
        session_date, files, lambda file_id: manifests[file_id]
    )
    assert report["expected_minutes"] == 375
    assert report["snapshot_files_present"] == 375
    assert report["minute_manifests_present"] == 375
    assert report["observed_quote_coverage_pct"] == 100.0
    assert report["quality_gate_pass"] is True
    assert report["audit_status"] == "PASS"
    assert len(rows) == 375


def test_audit_fails_when_a_minute_snapshot_is_missing():
    session_date = date(2026, 10, 16)
    files, manifests = _complete_day_files(session_date)
    missing_key = "20261016_1005"
    files = [
        item for item in files
        if item["name"] != f"snapshot_{missing_key}.parquet"
    ]
    report, _ = build_audit_report(
        session_date, files, lambda file_id: manifests[file_id]
    )
    assert report["quality_gate_pass"] is False
    assert missing_key in report["missing_snapshot_minutes"]



def test_audit_accepts_recovered_drive_write_when_artifacts_are_present():
    session_date = date(2026, 10, 16)
    files, manifests = _complete_day_files(session_date)
    recovered_key = "20261016_1005"
    recovered_id = f"manifest-{recovered_key}"
    manifests[recovered_id]["status"] = "PERSISTENCE_ERROR"

    report, _ = build_audit_report(
        session_date, files, lambda file_id: manifests[file_id]
    )
    assert report["quality_gate_pass"] is True
    assert recovered_key in report["recovered_persistence_issue_minutes"]
