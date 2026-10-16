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


def expected_minutes(session_date: date) -> list[str]:
    """Return 375 expected cash-session minute keys, 09:15 through 15:29 IST."""
    start = datetime.combine(session_date, time(9, 15), tzinfo=IST)
    return [
        (start + timedelta(minutes=offset)).strftime("%Y%m%d_%H%M")
        for offset in range(375)
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
) -> tuple[dict[str, Any], pd.DataFrame]:
    expected = expected_minutes(session_date)
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
        all_snapshots_present
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
    report, rows = build_audit_report(
        session_date,
        files,
        lambda file_id: _download_json(store, file_id),
    )

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
        description="Audit all-NSE-equity per-minute quote snapshots and Drive persistence."
    )
    parser.add_argument("--date", default="", help="NSE session date YYYY-MM-DD (default: today in IST)")
    parser.add_argument("--sync-drive", action="store_true", help="Upload audit JSON and CSV to Google Drive")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when the data-quality gate fails")
    args = parser.parse_args()
    session_date = date.fromisoformat(args.date) if args.date else datetime.now(IST).date()
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
