#!/usr/bin/env python3
"""Audit full-universe minute snapshots and per-minute Drive manifests."""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from market_session_guard import is_nse_trading_day

IST = ZoneInfo("Asia/Kolkata")
REMOTE_ROOT = "production_data/all_universe_minute_snapshots"
FOLDER_MIME = "application/vnd.google-apps.folder"
SNAPSHOT_PREFIX = "snapshot_"
MANIFEST_PREFIX = "manifest_"
CAPTURE_INTERVAL_MINUTES = 5
PREDICTION_ROOT = "production_data/all_nse_minute_predictions"


def expected_minutes(
    session_date: date, *, as_of: datetime | pd.Timestamp | None = None
) -> list[str]:
    """Return expected five-minute capture slots through the latest completed slot.

    A regular session has 75 scheduled slots: 09:15 through 15:25 IST.
    For a live session, do not mark future slots as missing; evaluate only slots
    whose scheduled minute has completed. Before the open, return no slots so the
    caller can select the latest completed session.
    """
    start = datetime.combine(session_date, time(9, 15), tzinfo=IST)
    end = datetime.combine(session_date, time(15, 25), tzinfo=IST)
    if as_of is not None:
        stamp = pd.Timestamp(as_of)
        stamp = stamp.tz_localize(IST) if stamp.tzinfo is None else stamp.tz_convert(IST)
        if session_date > stamp.date():
            return []
        if session_date == stamp.date():
            last_completed = stamp.floor("min") - pd.Timedelta(minutes=1)
            if last_completed < pd.Timestamp(start):
                return []
            end = min(end, last_completed.to_pydatetime())
    if end < start:
        return []
    count = int((end - start).total_seconds() // (60 * CAPTURE_INTERVAL_MINUTES)) + 1
    return [
        (start + timedelta(minutes=offset * CAPTURE_INTERVAL_MINUTES)).strftime("%Y%m%d_%H%M")
        for offset in range(count)
    ]


def _find_folder(store: Any, parts: list[str]) -> str | None:
    parent = store.root_id
    for part in parts:
        item = store._find_child(parent, part, FOLDER_MIME)
        if not item:
            return None
        parent = item["id"]
    return parent


def _list_children(store: Any, folder_id: str) -> list[dict[str, Any]]:
    page_token = None
    out: list[dict[str, Any]] = []
    while True:
        args: dict[str, Any] = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "spaces": "drive",
            "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime)",
            "pageSize": 1000,
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
        }
        if page_token:
            args["pageToken"] = page_token
        response = store.service.files().list(**args).execute()
        out.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return out


def _download_json(store: Any, file_id: str) -> dict[str, Any]:
    raw = store.service.files().get_media(fileId=file_id).execute()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = str(raw)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Per-minute manifest is not a JSON object")
    return value


def _group_by_name(files: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for item in files:
        result.setdefault(str(item.get("name") or ""), []).append(item)
    return result


def build_audit_report(
    session_date: date,
    files: list[dict[str, Any]],
    manifest_loader,
    *,
    as_of: datetime | pd.Timestamp | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    expected = expected_minutes(session_date, as_of=as_of)
    groups = _group_by_name(files)
    snapshot_files: dict[str, list[dict[str, Any]]] = {}
    minute_manifest_files: dict[str, list[dict[str, Any]]] = {}
    for key in expected:
        snapshot_files[key] = groups.get(f"snapshot_{key}.parquet", [])
        minute_manifest_files[key] = groups.get(f"manifest_{key}.json", [])

    rows: list[dict[str, Any]] = []
    corrupt_manifests: list[str] = []
    for key in expected:
        manifests = minute_manifest_files[key]
        manifest: dict[str, Any] = {}
        manifest_error = None
        if manifests:
            try:
                manifest = manifest_loader(manifests[0]["id"])
            except Exception as exc:
                manifest_error = f"{type(exc).__name__}: {exc}"
                corrupt_manifests.append(key)

        rows.append({
            "minute_key_ist": key,
            "snapshot_present": bool(snapshot_files[key]),
            "snapshot_duplicate_file_count": max(0, len(snapshot_files[key]) - 1),
            "manifest_present": bool(manifests),
            "manifest_duplicate_file_count": max(0, len(manifests) - 1),
            "capture_status": manifest.get("status", "MISSING_MANIFEST" if not manifests else "CORRUPT_MANIFEST"),
            "expected_symbol_count": manifest.get("expected_symbol_count"),
            "quoted_symbol_count": manifest.get("quoted_symbol_count"),
            "missing_quote_count": manifest.get("missing_quote_count"),
            "unmapped_symbol_count": manifest.get("unmapped_symbol_count"),
            "forecast_symbol_count": manifest.get("forecast_symbol_count"),
            "forecast_record_count": manifest.get("forecast_record_count"),
            "matched_outcome_count": manifest.get("matched_outcome_count"),
            "pending_snapshot_recovery_errors": len(
                (manifest.get("pending_snapshot_recovery") or {}).get("errors") or []
            ),
            "pending_manifest_recovery_errors": len(
                (manifest.get("pending_manifest_recovery") or {}).get("errors") or []
            ),
            "manifest_error": manifest_error,
        })

    frame = pd.DataFrame(rows)
    expected_count = len(expected)
    snapshots_present = int(frame["snapshot_present"].sum())
    manifests_present = int(frame["manifest_present"].sum())
    duplicate_snapshot_files = int(frame["snapshot_duplicate_file_count"].sum())
    duplicate_manifest_files = int(frame["manifest_duplicate_file_count"].sum())
    status_counts = {
        str(k): int(v)
        for k, v in frame["capture_status"].value_counts(dropna=False).items()
    }
    total_expected_symbols = pd.to_numeric(
        frame["expected_symbol_count"], errors="coerce"
    ).fillna(0).sum()
    total_quoted_symbols = pd.to_numeric(
        frame["quoted_symbol_count"], errors="coerce"
    ).fillna(0).sum()
    total_missing_quotes = pd.to_numeric(
        frame["missing_quote_count"], errors="coerce"
    ).fillna(0).sum()
    total_outcomes = pd.to_numeric(
        frame["matched_outcome_count"], errors="coerce"
    ).fillna(0).sum()
    all_snapshots_present = snapshots_present == expected_count
    all_manifests_present = manifests_present == expected_count
    recovered_persistence_issue_minutes = [
        str(row["minute_key_ist"])
        for row in rows
        if (
            row["capture_status"] == "PERSISTENCE_ERROR"
            and row["snapshot_present"]
            and row["manifest_present"]
            and pd.notna(row["expected_symbol_count"])
            and pd.notna(row["quoted_symbol_count"])
            and pd.notna(row["missing_quote_count"])
            and pd.notna(row["unmapped_symbol_count"])
            and int(row["expected_symbol_count"]) > 0
            and int(row["quoted_symbol_count"]) == int(row["expected_symbol_count"])
            and int(row["missing_quote_count"]) == 0
            and int(row["unmapped_symbol_count"]) == 0
        )
    ]
    statuses_data_complete = bool(all(
        row["capture_status"] == "OK"
        or str(row["minute_key_ist"]) in recovered_persistence_issue_minutes
        for row in rows
    ))
    coverage_pct = (
        float(total_quoted_symbols / total_expected_symbols * 100.0)
        if total_expected_symbols else 0.0
    )
    quality_gate_pass = bool(
        expected_count > 0
        and all_snapshots_present
        and all_manifests_present
        and statuses_data_complete
        and duplicate_snapshot_files == 0
        and duplicate_manifest_files == 0
        and int(total_missing_quotes) == 0
        and not corrupt_manifests
    )
    report = {
        "audit_name": "all-universe-minute-capture-audit",
        "schema_version": "all-universe-minute-audit-v1",
        "session_date": session_date.isoformat(),
        "timezone": "Asia/Kolkata",
        "nse_trading_day": is_nse_trading_day(session_date),
        "expected_minutes": expected_count,
        "snapshot_files_present": snapshots_present,
        "missing_snapshot_minutes": [
            key for key in expected if not snapshot_files[key]
        ],
        "minute_manifests_present": manifests_present,
        "missing_manifest_minutes": [
            key for key in expected if not minute_manifest_files[key]
        ],
        "corrupt_manifest_minutes": corrupt_manifests,
        "recovered_persistence_issue_minutes": recovered_persistence_issue_minutes,
        "duplicate_snapshot_files": duplicate_snapshot_files,
        "duplicate_manifest_files": duplicate_manifest_files,
        "capture_status_counts": status_counts,
        "total_expected_symbol_minutes": int(total_expected_symbols),
        "total_quoted_symbol_minutes": int(total_quoted_symbols),
        "total_missing_quote_rows": int(total_missing_quotes),
        "observed_quote_coverage_pct": round(coverage_pct, 4),
        "matched_outcomes": int(total_outcomes),
        "total_pending_snapshot_recovery_errors": int(
            pd.to_numeric(
                frame["pending_snapshot_recovery_errors"], errors="coerce"
            ).fillna(0).sum()
        ),
        "total_pending_manifest_recovery_errors": int(
            pd.to_numeric(
                frame["pending_manifest_recovery_errors"], errors="coerce"
            ).fillna(0).sum()
        ),
        "forecast_scope_note": (
            "Forecast counts describe only symbols for which the active forecast engine "
            "actually emitted rows; quote capture does not imply every symbol was forecast."
        ),
        "quote_timestamp_note": (
            "INDstocks quote timestamps are collector-observation times, not exchange-stamped "
            "one-minute OHLCV candle timestamps."
        ),
        "quality_gate_pass": quality_gate_pass,
        "audit_status": "PASS" if quality_gate_pass else "NEEDS_ATTENTION",
        "audit_note": (
            "PERSISTENCE_ERROR minutes are considered recovered only when both the "
            "snapshot and manifest now exist and quote coverage is complete; the original "
            "status remains visible in capture_status_counts."
        ),
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return report, frame


def build_prediction_audit(
    expected: list[str],
    files: list[dict[str, Any]],
    manifest_loader,
    *,
    session_date: date,
) -> dict[str, Any]:
    """Validate each scheduled all-NSE prediction Parquet and its manifest."""
    groups = _group_by_name(files)
    first_prediction_key = (
        datetime.combine(session_date, time(9, 15), tzinfo=IST) + timedelta(minutes=20)
    ).strftime("%Y%m%d_%H%M")
    missing_files: list[str] = []
    missing_manifests: list[str] = []
    corrupt_manifests: list[str] = []
    duplicate_files = 0
    duplicate_manifests = 0
    row_count_failures: list[str] = []
    model_quality_failures: list[str] = []
    total_expected_rows = 0
    total_persisted_rows = 0
    total_predictions = 0
    status_counts: dict[str, int] = {}

    for key in expected:
        snapshots = groups.get(f"predictions_{key}.parquet", [])
        manifests = groups.get(f"manifest_{key}.json", [])
        if not snapshots:
            missing_files.append(key)
        if not manifests:
            missing_manifests.append(key)
        duplicate_files += max(0, len(snapshots) - 1)
        duplicate_manifests += max(0, len(manifests) - 1)
        manifest: dict[str, Any] = {}
        if manifests:
            try:
                manifest = manifest_loader(manifests[0]["id"])
            except Exception:
                corrupt_manifests.append(key)
                continue
        status = str(manifest.get("status") or ("MISSING_MANIFEST" if not manifests else "CORRUPT_MANIFEST"))
        status_counts[status] = status_counts.get(status, 0) + 1
        expected_symbols = manifest.get("expected_symbol_count")
        row_count = manifest.get("row_count")
        prediction_count = manifest.get("prediction_count")
        try:
            expected_int = int(expected_symbols)
            row_count_int = int(row_count)
        except (TypeError, ValueError):
            expected_int = row_count_int = 0
        total_expected_rows += expected_int
        total_persisted_rows += row_count_int
        try:
            total_predictions += int(prediction_count or 0)
        except (TypeError, ValueError):
            pass
        if expected_int <= 0 or row_count_int != expected_int:
            row_count_failures.append(key)
        if key < first_prediction_key:
            continue  # Model history warm-up; the all-symbol rows must still be persisted.
        quality_ok = (
            status == "OK"
            and manifest.get("quality_gate_pass") is True
            and prediction_count is not None
            and int(prediction_count or 0) == expected_int
            and int(manifest.get("missing_quote_count") or 0) == 0
            and int(manifest.get("data_limited_count") or 0) == 0
            and int(manifest.get("untrained_symbol_count") or 0) == 0
            and int(manifest.get("model_failure_count") or 0) == 0
            and manifest.get("history_ready") is True
            and int(manifest.get("history_seed_coverage_symbols") or 0) == expected_int
            and int(manifest.get("history_seed_failed_batches") or 0) == 0
        )
        if not quality_ok:
            model_quality_failures.append(key)

    quality_gate_pass = bool(
        len(expected) > 0
        and len(missing_files) == 0
        and len(missing_manifests) == 0
        and duplicate_files == 0
        and duplicate_manifests == 0
        and len(corrupt_manifests) == 0
        and len(row_count_failures) == 0
        and len(model_quality_failures) == 0
    )
    return {
        "prediction_expected_slots": len(expected),
        "prediction_files_present": len(expected) - len(missing_files),
        "missing_prediction_minutes": missing_files,
        "prediction_manifests_present": len(expected) - len(missing_manifests),
        "missing_prediction_manifest_minutes": missing_manifests,
        "corrupt_prediction_manifest_minutes": corrupt_manifests,
        "duplicate_prediction_files": duplicate_files,
        "duplicate_prediction_manifests": duplicate_manifests,
        "prediction_row_count_failures": row_count_failures,
        "prediction_model_quality_failures": model_quality_failures,
        "prediction_status_counts": status_counts,
        "total_expected_prediction_symbol_rows": total_expected_rows,
        "total_persisted_prediction_rows": total_persisted_rows,
        "total_model_prediction_count": total_predictions,
        "prediction_warmup_until": first_prediction_key,
        "prediction_quality_gate_pass": quality_gate_pass,
    }


def run_audit(session_date: date, *, sync_drive: bool = True) -> tuple[dict[str, Any], pd.DataFrame]:
    from persistent_store import GoogleDriveStore

    store = GoogleDriveStore()
    day_parts = [
        "production_data",
        "all_universe_minute_snapshots",
        f"{session_date:%Y}",
        f"{session_date:%m}",
        f"{session_date:%d}",
    ]
    day_folder_id = _find_folder(store, day_parts)
    files = _list_children(store, day_folder_id) if day_folder_id else []
    load_manifest = lambda file_id: _download_json(store, file_id)
    as_of = datetime.now(IST)
    report, rows = build_audit_report(
        session_date,
        files,
        load_manifest,
        as_of=as_of,
    )
    prediction_day_parts = [
        "production_data",
        "all_nse_minute_predictions",
        f"{session_date:%Y}",
        f"{session_date:%m}",
        f"{session_date:%d}",
    ]
    prediction_folder_id = _find_folder(store, prediction_day_parts)
    prediction_files = _list_children(store, prediction_folder_id) if prediction_folder_id else []
    prediction_report = build_prediction_audit(
        expected_minutes(session_date, as_of=as_of),
        prediction_files,
        load_manifest,
        session_date=session_date,
    )
    report.update(prediction_report)
    report["quality_gate_pass"] = bool(
        report.get("quality_gate_pass") and prediction_report["prediction_quality_gate_pass"]
    )
    report["audit_status"] = "PASS" if report["quality_gate_pass"] else "NEEDS_ATTENTION"

    output_dir = (
        Path("data")
        / "all_universe_minute_audits"
        / f"{session_date:%Y}"
        / f"{session_date:%m}"
        / f"{session_date:%d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"all_universe_minute_audit_{session_date:%Y%m%d}.json"
    csv_path = output_dir / f"all_universe_minute_audit_{session_date:%Y%m%d}.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    rows.to_csv(csv_path, index=False)
    report["local_json_path"] = str(json_path)
    report["local_csv_path"] = str(csv_path)

    if sync_drive:
        remote_base = (
            "production_data/all_universe_minute_snapshots/audits/"
            f"{session_date:%Y/%m/%d}"
        )
        report["drive_json_write"] = store.upload_or_update(
            json_path,
            f"{remote_base}/{json_path.name}",
            mime_type="application/json",
        )
        report["drive_csv_write"] = store.upload_or_update(
            csv_path,
            f"{remote_base}/{csv_path.name}",
            mime_type="text/csv",
        )
        if (
            report["drive_json_write"].get("status") != "OK"
            or report["drive_csv_write"].get("status") != "OK"
        ):
            report["audit_status"] = "PERSISTENCE_ERROR"
            report["quality_gate_pass"] = False
        # Include write results in the final JSON stored on local disk and, if
        # possible, overwrite the first Drive JSON so it contains final status.
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if report.get("drive_json_write", {}).get("status") == "OK":
            final_json_write = store.upload_or_update(
                json_path,
                f"{remote_base}/{json_path.name}",
                mime_type="application/json",
            )
            report["drive_json_write_final"] = final_json_write

    return report, rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit all-NSE-equity five-minute quote and prediction snapshots plus Drive persistence."
    )
    parser.add_argument("--date", default="", help="NSE session date YYYY-MM-DD (default: today in IST)")
    parser.add_argument("--sync-drive", action="store_true", help="Upload audit JSON and CSV to Google Drive")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when the data-quality gate fails")
    args = parser.parse_args()
    if args.date:
        session_date = date.fromisoformat(args.date)
    else:
        now = datetime.now(IST)
        session_date = now.date()
        if now.time() < time(9, 15):
            session_date -= timedelta(days=1)
        while not is_nse_trading_day(session_date):
            session_date -= timedelta(days=1)
    if not is_nse_trading_day(session_date):
        print(json.dumps({
            "session_date": session_date.isoformat(),
            "status": "SKIPPED_NON_TRADING_DAY",
        }, sort_keys=True))
        return 0

    report, _ = run_audit(session_date, sync_drive=args.sync_drive)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("quality_gate_pass") or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
