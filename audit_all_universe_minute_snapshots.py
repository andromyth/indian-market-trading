#!/usr/bin/env python3
"""Audit full-universe minute snapshots and per-minute Drive manifests."""
from __future__ import annotations

import argparse
import io
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



def _snapshot_completeness(frame: pd.DataFrame) -> dict[str, Any]:
    """Classify complete and incomplete quote rows independently by symbol."""
    required = [
        "symbol",
        "instrument_code",
        "quote_status",
        "actual_observed_price",
        "quote_observed_at_utc",
    ]
    empty_result = {
        "snapshot_row_count": 0,
        "complete_symbol_count": 0,
        "incomplete_symbol_count": 0,
        "snapshot_duplicate_symbol_count": 0,
        "symbol_capture_status_counts": {},
        "incomplete_symbol_samples": [],
        "missing_required_snapshot_columns": [],
        "snapshot_error": None,
    }
    if not isinstance(frame, pd.DataFrame):
        return {
            **empty_result,
            "missing_required_snapshot_columns": required,
            "snapshot_error": "Snapshot loader did not return a DataFrame",
        }
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        return {
            **empty_result,
            "snapshot_row_count": int(len(frame)),
            "missing_required_snapshot_columns": missing_columns,
            "snapshot_error": "Snapshot is missing required core quote columns",
        }
    if frame.empty:
        return {
            **empty_result,
            "snapshot_error": "Snapshot contains no symbol rows",
        }

    symbols = frame["symbol"].fillna("").astype(str).str.strip().str.upper()
    nonblank = symbols.ne("")
    symbol_counts = symbols.loc[nonblank].value_counts()
    duplicate_symbol_count = int(
        sum(int(count) - 1 for count in symbol_counts.values if int(count) > 1)
    )
    unique_symbol = symbols.map(symbol_counts).fillna(0).eq(1)
    statuses = frame["quote_status"].fillna("").astype(str).str.strip().str.upper()
    instruments = frame["instrument_code"].fillna("").astype(str).str.strip()
    prices = pd.to_numeric(frame["actual_observed_price"], errors="coerce")
    price_valid = prices.gt(0) & prices.lt(float("inf"))
    timestamps = pd.to_datetime(
        frame["quote_observed_at_utc"], errors="coerce", utc=True
    )
    complete = (
        nonblank
        & unique_symbol
        & statuses.eq("PASS")
        & instruments.ne("")
        & price_valid
        & timestamps.notna()
    )

    incomplete_samples = []
    for index in frame.index[~complete]:
        reasons = []
        symbol = str(symbols.loc[index] or "")
        if not symbol:
            reasons.append("MISSING_SYMBOL")
        elif not bool(unique_symbol.loc[index]):
            reasons.append("DUPLICATE_SYMBOL")
        if str(statuses.loc[index]) != "PASS":
            reasons.append(str(statuses.loc[index]) or "MISSING_QUOTE_STATUS")
        if not str(instruments.loc[index]):
            reasons.append("MISSING_INSTRUMENT_CODE")
        if not bool(price_valid.loc[index]):
            reasons.append("INVALID_OR_MISSING_PRICE")
        if pd.isna(timestamps.loc[index]):
            reasons.append("INVALID_OR_MISSING_QUOTE_TIMESTAMP")
        if len(incomplete_samples) < 100:
            incomplete_samples.append({
                "symbol": symbol or None,
                "reasons": reasons,
            })

    return {
        "snapshot_row_count": int(len(frame)),
        "complete_symbol_count": int(complete.sum()),
        "incomplete_symbol_count": int((~complete).sum()),
        "snapshot_duplicate_symbol_count": duplicate_symbol_count,
        "symbol_capture_status_counts": {
            str(key): int(value)
            for key, value in statuses.value_counts(dropna=False).items()
        },
        "incomplete_symbol_samples": incomplete_samples,
        "missing_required_snapshot_columns": [],
        "snapshot_error": None,
    }


def build_audit_report(
    session_date: date,
    files: list[dict[str, Any]],
    manifest_loader,
    *,
    snapshot_loader,
    as_of: datetime | pd.Timestamp | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Record five-minute capture diagnostics without a universe-wide pass/fail gate."""
    expected = expected_minutes(session_date, as_of=as_of)
    groups = _group_by_name(files)
    snapshot_files: dict[str, list[dict[str, Any]]] = {}
    minute_manifest_files: dict[str, list[dict[str, Any]]] = {}
    for key in expected:
        snapshot_files[key] = groups.get(f"snapshot_{key}.parquet", [])
        minute_manifest_files[key] = groups.get(f"manifest_{key}.json", [])

    rows: list[dict[str, Any]] = []
    corrupt_manifests: list[str] = []
    corrupt_snapshots: list[str] = []
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

        snapshot_metrics = {
            "snapshot_row_count": 0,
            "complete_symbol_count": 0,
            "incomplete_symbol_count": 0,
            "snapshot_duplicate_symbol_count": 0,
            "symbol_capture_status_counts": {},
            "incomplete_symbol_samples": [],
            "missing_required_snapshot_columns": [],
            "snapshot_error": "Snapshot file is missing",
        }
        if snapshot_files[key]:
            try:
                snapshot_frame = snapshot_loader(snapshot_files[key][0]["id"])
                snapshot_metrics = _snapshot_completeness(snapshot_frame)
                if snapshot_metrics.get("snapshot_error"):
                    corrupt_snapshots.append(key)
            except Exception as exc:
                snapshot_metrics["snapshot_error"] = f"{type(exc).__name__}: {exc}"
                corrupt_snapshots.append(key)
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
            **snapshot_metrics,
        })

    frame = pd.DataFrame(rows)
    expected_count = len(expected)
    snapshots_present = int(frame["snapshot_present"].sum()) if not frame.empty else 0
    manifests_present = int(frame["manifest_present"].sum()) if not frame.empty else 0
    duplicate_snapshot_files = int(frame["snapshot_duplicate_file_count"].sum()) if not frame.empty else 0
    duplicate_manifest_files = int(frame["manifest_duplicate_file_count"].sum()) if not frame.empty else 0
    status_counts = {
        str(key): int(value)
        for key, value in frame["capture_status"].value_counts(dropna=False).items()
    } if not frame.empty else {}
    total_expected_symbols = pd.to_numeric(
        frame["expected_symbol_count"], errors="coerce"
    ).fillna(0).sum() if not frame.empty else 0
    total_quoted_symbols = pd.to_numeric(
        frame["quoted_symbol_count"], errors="coerce"
    ).fillna(0).sum() if not frame.empty else 0
    total_missing_quotes = pd.to_numeric(
        frame["missing_quote_count"], errors="coerce"
    ).fillna(0).sum() if not frame.empty else 0
    total_outcomes = pd.to_numeric(
        frame["matched_outcome_count"], errors="coerce"
    ).fillna(0).sum() if not frame.empty else 0
    coverage_pct = (
        float(total_quoted_symbols / total_expected_symbols * 100.0)
        if total_expected_symbols else 0.0
    )
    frame["complete_symbol_coverage_pct"] = frame.apply(
        lambda row: (
            float(row["complete_symbol_count"] / row["expected_symbol_count"] * 100.0)
            if pd.notna(row["expected_symbol_count"])
            and float(row["expected_symbol_count"]) > 0
            else 0.0
        ),
        axis=1,
    ) if not frame.empty else pd.Series(dtype=float)
    slot_coverages = (
        pd.to_numeric(frame["complete_symbol_coverage_pct"], errors="coerce").fillna(0).tolist()
        if not frame.empty else []
    )
    minimum_complete_symbol_coverage_pct = min(slot_coverages) if slot_coverages else 0.0
    average_complete_symbol_coverage_pct = (
        sum(slot_coverages) / len(slot_coverages) if slot_coverages else 0.0
    )
    total_complete_symbols = int(
        pd.to_numeric(frame["complete_symbol_count"], errors="coerce").fillna(0).sum()
    ) if not frame.empty else 0
    slots_with_incomplete_symbols = [
        str(row["minute_key_ist"])
        for _, row in frame.iterrows()
        if (
            pd.notna(row.get("expected_symbol_count"))
            and int(row.get("complete_symbol_count") or 0)
                < int(row.get("expected_symbol_count") or 0)
        )
    ]
    recovered_persistence_issue_minutes = [
        str(row["minute_key_ist"])
        for _, row in frame.iterrows()
        if (
            row["capture_status"] == "PERSISTENCE_ERROR"
            and row["snapshot_present"]
            and row["manifest_present"]
            and not row.get("snapshot_error")
            and int(row.get("complete_symbol_count") or 0) > 0
        )
    ]
    report = {
        "audit_name": "all-universe-minute-capture-audit",
        "schema_version": "all-universe-minute-audit-v4",
        "session_date": session_date.isoformat(),
        "timezone": "Asia/Kolkata",
        "capture_interval_minutes": CAPTURE_INTERVAL_MINUTES,
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
        "corrupt_snapshot_minutes": corrupt_snapshots,
        "slots_with_incomplete_symbols": slots_with_incomplete_symbols,
        "recovered_persistence_issue_minutes": recovered_persistence_issue_minutes,
        "duplicate_snapshot_files": duplicate_snapshot_files,
        "duplicate_manifest_files": duplicate_manifest_files,
        "capture_status_counts": status_counts,
        "total_expected_symbol_minutes": int(total_expected_symbols),
        "total_quoted_symbol_minutes": int(total_quoted_symbols),
        "total_missing_quote_rows": int(total_missing_quotes),
        "observed_quote_coverage_pct": round(coverage_pct, 4),
        "total_complete_symbol_rows": total_complete_symbols,
        "average_complete_symbol_coverage_pct": round(average_complete_symbol_coverage_pct, 4),
        "minimum_complete_symbol_coverage_pct": round(minimum_complete_symbol_coverage_pct, 4),
        "coverage_is_informational": True,
        "matched_outcomes": int(total_outcomes),
        "total_pending_snapshot_recovery_errors": int(
            pd.to_numeric(
                frame["pending_snapshot_recovery_errors"], errors="coerce"
            ).fillna(0).sum()
        ) if not frame.empty else 0,
        "total_pending_manifest_recovery_errors": int(
            pd.to_numeric(
                frame["pending_manifest_recovery_errors"], errors="coerce"
            ).fillna(0).sum()
        ) if not frame.empty else 0,
        "forecast_scope_note": (
            "Forecast counts describe only symbols for which the active forecast engine "
            "actually emitted rows; quote capture does not imply every symbol was forecast."
        ),
        "quote_timestamp_note": (
            "INDstocks quote timestamps are collector-observation times, not exchange-stamped "
            "one-minute OHLCV candle timestamps."
        ),
        "symbol_level_policy": (
            "Each symbol with one complete same-slot quote row is independently eligible for "
            "prediction after the stored snapshot is confirmed. Missing or incomplete symbols "
            "do not block predictions for other symbols."
        ),
        "audit_status": "RECORDED",
        "audit_note": (
            "This report records slot and per-symbol diagnostics only. Coverage percentages, "
            "missing symbols, partial slots and model-readiness counts are informational; "
            "there is no universe-wide or percentage-based acceptance threshold."
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
    """Record prediction storage and same-slot linkage without percentage gates."""
    groups = _group_by_name(files)
    first_prediction_key = (
        datetime.combine(session_date, time(9, 15), tzinfo=IST) + timedelta(minutes=20)
    ).strftime("%Y%m%d_%H%M")
    missing_files: list[str] = []
    missing_manifests: list[str] = []
    corrupt_manifests: list[str] = []
    duplicate_files = 0
    duplicate_manifests = 0
    row_count_errors: list[str] = []
    snapshot_link_diagnostics: list[str] = []
    model_diagnostics: list[dict[str, Any]] = []
    total_declared_rows = 0
    total_predictions = 0
    status_counts: dict[str, int] = {}
    prediction_status_counts: dict[str, int] = {}

    for key in expected:
        snapshots = groups.get(f"predictions_{key}.parquet", [])
        manifests = groups.get(f"manifest_{key}.json", [])
        if not snapshots:
            missing_files.append(key)
        if not manifests:
            missing_manifests.append(key)
        duplicate_files += max(0, len(snapshots) - 1)
        duplicate_manifests += max(0, len(manifests) - 1)
        if not manifests:
            continue
        try:
            manifest = manifest_loader(manifests[0]["id"])
        except Exception:
            corrupt_manifests.append(key)
            continue

        status = str(manifest.get("status") or "UNKNOWN").upper()
        status_counts[status] = status_counts.get(status, 0) + 1
        try:
            expected_symbols = int(manifest.get("expected_symbol_count") or 0)
            row_count = int(manifest.get("row_count") or 0)
            prediction_count = int(manifest.get("prediction_count") or 0)
        except (TypeError, ValueError):
            expected_symbols = row_count = prediction_count = -1
        if min(expected_symbols, row_count, prediction_count) < 0:
            row_count_errors.append(key)
            continue
        total_declared_rows += row_count
        total_predictions += prediction_count
        for status_name, count in (manifest.get("prediction_status_counts") or {}).items():
            prediction_status_counts[str(status_name)] = (
                prediction_status_counts.get(str(status_name), 0) + int(count or 0)
            )

        expected_snapshot_minute = datetime.strptime(key, "%Y%m%d_%H%M").replace(tzinfo=IST)
        expected_snapshot_path = (
            "production_data/all_universe_minute_snapshots/"
            f"{session_date:%Y/%m/%d}/snapshot_{key}.parquet"
        )
        snapshot_link_ok = (
            bool(snapshots)
            and manifest.get("quote_snapshot_rows_validated") is True
            and manifest.get("quote_snapshot_cycle_minute_ist") == expected_snapshot_minute.isoformat()
            and manifest.get("quote_snapshot_drive_path") == expected_snapshot_path
            and manifest.get("quote_snapshot_expected_symbol_count") == expected_symbols
            and manifest.get("quote_snapshot_capture_status") in {"OK", "PARTIAL", "STORED"}
        )
        if not snapshot_link_ok:
            snapshot_link_diagnostics.append(key)

        if key >= first_prediction_key and (
            prediction_count == 0
            or int(manifest.get("model_failure_count") or 0) > 0
            or manifest.get("model_artifact_verified") is not True
        ):
            model_diagnostics.append({
                "minute_key_ist": key,
                "prediction_count": prediction_count,
                "stored_rows": row_count,
                "eligible_symbol_count": manifest.get("eligible_symbol_count"),
                "model_failure_count": manifest.get("model_failure_count"),
                "data_limited_count": manifest.get("data_limited_count"),
                "untrained_symbol_count": manifest.get("untrained_symbol_count"),
                "model_artifact_verified": manifest.get("model_artifact_verified"),
                "model_status": manifest.get("model_status"),
                "prediction_status_counts": manifest.get("prediction_status_counts") or {},
            })

    return {
        "prediction_expected_slots": len(expected),
        "prediction_files_present": len(expected) - len(missing_files),
        "missing_prediction_minutes": missing_files,
        "prediction_manifests_present": len(expected) - len(missing_manifests),
        "missing_prediction_manifest_minutes": missing_manifests,
        "corrupt_prediction_manifest_minutes": corrupt_manifests,
        "duplicate_prediction_files": duplicate_files,
        "duplicate_prediction_manifests": duplicate_manifests,
        "prediction_row_count_errors": row_count_errors,
        "prediction_snapshot_link_diagnostics": snapshot_link_diagnostics,
        "prediction_model_diagnostics": model_diagnostics,
        "prediction_status_counts": prediction_status_counts,
        "prediction_slot_status_counts": status_counts,
        "total_declared_prediction_rows": total_declared_rows,
        "total_model_prediction_count": total_predictions,
        "prediction_warmup_until": first_prediction_key,
        "prediction_coverage_is_informational": True,
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

    def load_snapshot(file_id: str) -> pd.DataFrame:
        raw = store.service.files().get_media(fileId=file_id).execute()
        if hasattr(raw, "read"):
            raw = raw.read()
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("Google Drive snapshot download did not return Parquet bytes")
        return pd.read_parquet(io.BytesIO(raw))

    report, rows = build_audit_report(
        session_date,
        files,
        load_manifest,
        snapshot_loader=load_snapshot,
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
    report["audit_status"] = "RECORDED"

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
    parser.add_argument("--strict", action="store_true", help="Deprecated; retained for compatibility and does not apply a global quality gate.")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
