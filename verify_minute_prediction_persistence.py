#!/usr/bin/env python3
"""Read-only Google Drive audit for minute-level prediction/actual persistence.

Prints counts, timestamps, health flags and remote paths only. It does not
modify files, write to Drive, retrain the model, or place orders.
"""
from __future__ import annotations

import io
import json
import os
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from persistent_store import GoogleDriveStore, drive_status

IST = ZoneInfo("Asia/Kolkata")
FOLDER_MIME = "application/vnd.google-apps.folder"


def _safe_error(exc: Exception) -> str:
    # Avoid dumping credential-bearing objects or HTTP request bodies to logs.
    return f"{type(exc).__name__}: {str(exc)[:240]}"


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _age_minutes(value: Any, now_utc: datetime) -> float | None:
    dt = _parse_datetime(value)
    if dt is None:
        return None
    return round(max(0.0, (now_utc - dt).total_seconds()) / 60.0, 2)


def _list_children(store: GoogleDriveStore, folder_id: str) -> list[dict]:
    result: list[dict] = []
    page_token = None
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
        result.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return result


def _folder_id(store: GoogleDriveStore, remote_folder: str) -> str | None:
    parent = store.root_id
    parts = [p for p in remote_folder.strip("/").split("/") if p]
    for part in parts:
        hit = store._find_child(parent, part, FOLDER_MIME)
        if not hit:
            return None
        parent = hit["id"]
    return parent


def _list_path(store: GoogleDriveStore, remote_folder: str) -> list[dict] | None:
    folder_id = _folder_id(store, remote_folder)
    if not folder_id:
        return None
    return _list_children(store, folder_id)


def _read_file_bytes(store: GoogleDriveStore, file_id: str) -> bytes:
    from googleapiclient.http import MediaIoBaseDownload

    request = store.service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


def _find_file(store: GoogleDriveStore, remote_path: str) -> dict | None:
    parts = [p for p in remote_path.strip("/").split("/") if p]
    if not parts:
        return None
    parent_path, filename = "/".join(parts[:-1]), parts[-1]
    siblings = _list_path(store, parent_path)
    if siblings is None:
        return None
    return next((x for x in siblings if x.get("name") == filename), None)


def _read_json(store: GoogleDriveStore, remote_path: str) -> tuple[dict | None, dict | None]:
    meta = _find_file(store, remote_path)
    if not meta:
        return None, None
    try:
        payload = json.loads(_read_file_bytes(store, meta["id"]).decode("utf-8"))
        return (payload if isinstance(payload, dict) else {"_non_object_json": True}), meta
    except Exception as exc:
        return {"_read_error": _safe_error(exc)}, meta


def _read_jsonl(store: GoogleDriveStore, file_meta: dict, max_bytes: int = 30_000_000) -> dict:
    size = int(file_meta.get("size") or 0)
    if size > max_bytes:
        return {"read": False, "reason": "ledger exceeds safe audit download limit", "size_bytes": size}
    raw = _read_file_bytes(store, file_meta["id"])
    rows = []
    malformed = 0
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
            else:
                malformed += 1
        except json.JSONDecodeError:
            malformed += 1
    symbols = set()
    horizons = set()
    latest_decision = None
    prediction_count = 0
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            symbols.add(symbol)
        horizon = row.get("horizon_min")
        if horizon is not None:
            horizons.add(str(horizon))
        if row.get("horizon_min") is not None or row.get("predicted_price") is not None or row.get("probability_up") is not None:
            prediction_count += 1
        for key in ("decision_timestamp", "worker_observed_at_utc", "created_at", "timestamp"):
            dt = _parse_datetime(row.get(key))
            if dt and (latest_decision is None or dt > latest_decision):
                latest_decision = dt
    return {
        "read": True,
        "file_bytes": len(raw),
        "rows": len(rows),
        "malformed_rows": malformed,
        "symbols": len(symbols),
        "horizons_minutes": sorted(horizons, key=lambda x: float(x) if x.replace(".", "", 1).isdigit() else 9999),
        "prediction_rows": prediction_count,
        "latest_prediction_timestamp_utc": latest_decision.isoformat() if latest_decision else None,
    }


def _snapshot_summary(
    store: GoogleDriveStore,
    files: list[dict],
    now_utc: datetime,
    session_date: date,
) -> dict:
    """Audit each expected one-minute snapshot and its actual/prediction content."""
    from market_session_guard import CLOSE, OPEN, is_nse_trading_day

    snapshots = sorted(
        [
            f for f in files
            if re.fullmatch(r"snapshot_\d{8}_\d{4}\.jsonl", str(f.get("name", "")))
        ],
        key=lambda x: x["name"],
    )
    by_minute: dict[str, list[dict]] = {}
    for item in snapshots:
        match = re.fullmatch(r"snapshot_(\d{8}_\d{4})\.jsonl", str(item.get("name", "")))
        if match:
            by_minute.setdefault(match.group(1), []).append(item)

    now_ist = now_utc.astimezone(IST)
    expected_keys: list[str] = []
    if is_nse_trading_day(session_date):
        start = datetime.combine(session_date, OPEN, tzinfo=IST)
        end = datetime.combine(session_date, time(CLOSE.hour, CLOSE.minute - 1), tzinfo=IST)
        if session_date == now_ist.date():
            last_completed_minute = now_ist.replace(second=0, microsecond=0) - timedelta(minutes=1)
            end = min(end, last_completed_minute)
        cursor = start
        while cursor <= end:
            expected_keys.append(cursor.strftime("%Y%m%d_%H%M"))
            cursor += timedelta(minutes=1)

    missing = [key for key in expected_keys if not by_minute.get(key)]
    duplicates = [key for key in expected_keys if len(by_minute.get(key, [])) > 1]
    opening_key = datetime.combine(session_date, OPEN, tzinfo=IST).strftime("%Y%m%d_%H%M")
    required_forecast_keys = [key for key in expected_keys if key != opening_key]

    summary = {
        "file_count_today": len(snapshots),
        "latest_file": snapshots[-1].get("name") if snapshots else None,
        "latest_file_modified_time_utc": snapshots[-1].get("modifiedTime") if snapshots else None,
        "latest_file_modified_age_minutes": _age_minutes(snapshots[-1].get("modifiedTime"), now_utc) if snapshots else None,
        "latest_file_timestamp_age_minutes": None,
        "expected_snapshot_minutes": len(expected_keys),
        "expected_prediction_actual_minutes": len(required_forecast_keys),
        "missing_snapshot_minutes": missing,
        "duplicate_snapshot_minutes": duplicates,
        "minutes_with_complete_prediction_and_actual": 0,
        "incomplete_prediction_actual_minutes": [],
        "prediction_rows_all_expected_minutes": 0,
        "actual_symbol_observation_rows_all_expected_minutes": 0,
        "symbol_observation_rows_all_expected_minutes": 0,
        "opening_minute_prediction_exempt": opening_key in expected_keys,
        "latest_snapshot_rows": 0,
        "latest_rows_with_actual_observed_close": 0,
        "latest_rows_with_predictions": 0,
        "latest_prediction_rows": 0,
        "latest_symbols_with_predictions": 0,
        "latest_quality_status_counts": {},
        "recent_5_snapshot_files_prediction_rows": 0,
        "recent_5_snapshot_files_actual_rows": 0,
        "snapshot_read_errors": [],
        "read_error": None,
    }
    latest = snapshots[-1] if snapshots else None
    if latest:
        match = re.fullmatch(r"snapshot_(\d{8})_(\d{4})\.jsonl", latest["name"])
        if match:
            try:
                local_dt = datetime.strptime(
                    match.group(1) + match.group(2), "%Y%m%d%H%M"
                ).replace(tzinfo=IST)
                summary["latest_file_timestamp_age_minutes"] = _age_minutes(local_dt.isoformat(), now_utc)
            except ValueError:
                pass

    latest_name = latest.get("name") if latest else None
    symbol_predictions: set[str] = set()
    recent_names = {f.get("name") for f in snapshots[-5:]}
    for key in expected_keys:
        matching_files = by_minute.get(key, [])
        if not matching_files:
            if key in required_forecast_keys:
                summary["incomplete_prediction_actual_minutes"].append(key)
            continue
        if len(matching_files) != 1:
            if key in required_forecast_keys:
                summary["incomplete_prediction_actual_minutes"].append(key)
            continue
        item = matching_files[0]
        try:
            raw = _read_file_bytes(store, item["id"])
            rows: list[dict] = []
            malformed_rows = 0
            for line in raw.decode("utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                    else:
                        malformed_rows += 1
                except json.JSONDecodeError:
                    malformed_rows += 1
            if malformed_rows:
                summary["snapshot_read_errors"].append({
                    "minute": key,
                    "error": f"{malformed_rows} malformed/non-object JSONL row(s)",
                })
            prediction_rows = sum(len(row.get("predictions") or []) for row in rows)
            symbol_rows = [row for row in rows if str(row.get("symbol") or "").strip()]
            actual_rows = [
                row for row in symbol_rows
                if row.get("actual_observed_close") is not None
                and (row.get("actual_observed_timestamp_ist") or row.get("actual_observed_timestamp_utc"))
            ]
            summary["prediction_rows_all_expected_minutes"] += prediction_rows
            summary["actual_symbol_observation_rows_all_expected_minutes"] += len(actual_rows)
            summary["symbol_observation_rows_all_expected_minutes"] += len(symbol_rows)
            complete = bool(symbol_rows) and all(
                str(row.get("data_quality_status") or "").upper() == "PASS"
                and row.get("actual_observed_close") is not None
                and (row.get("actual_observed_timestamp_ist") or row.get("actual_observed_timestamp_utc"))
                and bool(row.get("predictions"))
                for row in symbol_rows
            )
            if key in required_forecast_keys:
                if complete:
                    summary["minutes_with_complete_prediction_and_actual"] += 1
                else:
                    summary["incomplete_prediction_actual_minutes"].append(key)

            if item.get("name") in recent_names:
                summary["recent_5_snapshot_files_prediction_rows"] += prediction_rows
                summary["recent_5_snapshot_files_actual_rows"] += len(actual_rows)

            if item.get("name") == latest_name:
                quality: dict[str, int] = {}
                for row in rows:
                    label = str(row.get("data_quality_status") or "UNKNOWN")
                    quality[label] = quality.get(label, 0) + 1
                    if row.get("actual_observed_close") is not None:
                        summary["latest_rows_with_actual_observed_close"] += 1
                    preds = row.get("predictions") or []
                    if preds:
                        summary["latest_rows_with_predictions"] += 1
                        if row.get("symbol"):
                            symbol_predictions.add(str(row["symbol"]).upper())
                    summary["latest_prediction_rows"] += len(preds)
                summary["latest_snapshot_rows"] = len(rows)
                summary["latest_symbols_with_predictions"] = len(symbol_predictions)
                summary["latest_quality_status_counts"] = quality
        except Exception as exc:
            error = _safe_error(exc)
            summary["snapshot_read_errors"].append({"minute": key, "error": error})
            if key in required_forecast_keys:
                summary["incomplete_prediction_actual_minutes"].append(key)
            if item.get("name") == latest_name:
                summary["read_error"] = error

    summary["all_expected_snapshot_files_present"] = (
        len(expected_keys) > 0 and not missing and not duplicates
    )
    summary["all_required_minutes_have_predictions_and_actuals"] = (
        len(required_forecast_keys) > 0
        and summary["minutes_with_complete_prediction_and_actual"] == len(required_forecast_keys)
        and not summary["snapshot_read_errors"]
    )
    return summary

def _manifest_summary(store: GoogleDriveStore, path: str, now_utc: datetime) -> dict:
    payload, meta = _read_json(store, path)
    if payload is None:
        return {"exists": False, "path": path}
    if "_read_error" in payload:
        return {"exists": True, "path": path, "read_error": payload["_read_error"]}
    return {
        "exists": True,
        "path": path,
        "modified_time_utc": meta.get("modifiedTime") if meta else None,
        "modified_age_minutes": _age_minutes(meta.get("modifiedTime"), now_utc) if meta else None,
        "cycle_minute_ist": payload.get("cycle_minute_ist"),
        "cycle_minute_age_minutes": _age_minutes(payload.get("cycle_minute_ist"), now_utc),
        "status": payload.get("status"),
        "expected_symbol_count": payload.get("expected_symbol_count"),
        "quoted_symbol_count": payload.get("quoted_symbol_count"),
        "missing_quote_count": payload.get("missing_quote_count"),
        "unmapped_symbol_count": payload.get("unmapped_symbol_count"),
        "forecast_record_count": payload.get("forecast_record_count"),
        "matched_outcome_count": payload.get("matched_outcome_count"),
        "row_count": payload.get("row_count"),
        "snapshot_drive_path": payload.get("snapshot_drive_path"),
        "error": str(payload.get("error") or payload.get("manifest_sync_error") or "")[:240] or None,
    }


def main() -> int:
    now = datetime.now(IST)
    now_utc = now.astimezone(timezone.utc)
    today = now.date()

    drive = drive_status(require_write=True)
    drive_safe = {
        key: drive.get(key)
        for key in (
            "configured", "reachable", "write_ready", "auth_method",
            "write_auth_method", "oauth_refresh_token_present", "reason",
        )
    }
    print("DRIVE_PREFLIGHT:", json.dumps(drive_safe, sort_keys=True))
    if not drive.get("configured") or drive.get("reachable") is False or drive.get("write_ready") is False:
        print(json.dumps({"verdict": "DRIVE_PREFLIGHT_FAILED", "checked_at_ist": now.isoformat()}, indent=2))
        return 2

    store = GoogleDriveStore()
    watch_snapshots: dict[str, dict] = {}
    minute_rows: dict[str, dict] = {}
    ledgers: dict[str, dict] = {}
    bars: dict[str, dict] = {}
    all_universe: dict[str, dict] = {}
    outcomes: dict[str, dict] = {}

    try:
        lookback_days = int(os.getenv("MINUTE_PERSISTENCE_AUDIT_LOOKBACK_DAYS", "7"))
    except (TypeError, ValueError):
        lookback_days = 7
    lookback_days = max(1, min(7, lookback_days))

    for offset in range(lookback_days):
        day = today - timedelta(days=offset)
        day_key = day.isoformat()
        y, m, d = f"{day:%Y}", f"{day:%m}", f"{day:%d}"

        watch_path = f"production_data/intraday_minute_snapshots/{y}/{m}/{d}"
        watch_files = _list_path(store, watch_path)
        watch_files = watch_files or []
        watch_snapshots[day_key] = _snapshot_summary(store, watch_files, now_utc, day)

        ledger_path = f"production_data/intraday_predictions/{y}/{m}/{d}"
        ledger_files = _list_path(store, ledger_path) or []
        ledger_meta = next((f for f in ledger_files if f.get("name") == "predictions_service.jsonl"), None)
        if ledger_meta:
            try:
                detail = _read_jsonl(store, ledger_meta)
            except Exception as exc:
                detail = {"read": False, "read_error": _safe_error(exc), "file_bytes": int(ledger_meta.get("size") or 0)}
            ledgers[day_key] = {
                "exists": True,
                "file": ledger_meta.get("name"),
                "modified_time_utc": ledger_meta.get("modifiedTime"),
                "modified_age_minutes": _age_minutes(ledger_meta.get("modifiedTime"), now_utc),
                **detail,
            }
        else:
            ledgers[day_key] = {"exists": False}

        bar_path = f"production_data/intraday_live_worker/{y}/{m}/{d}"
        bar_files = _list_path(store, bar_path) or []
        bar_files = [f for f in bar_files if str(f.get("name", "")).lower().endswith(".parquet")]
        bars[day_key] = {
            "parquet_partition_files": len(bar_files),
            "latest_modified_time_utc": max((f.get("modifiedTime", "") for f in bar_files), default=None) or None,
        }

        universe_path = f"production_data/all_universe_minute_snapshots/{y}/{m}/{d}"
        universe_files = _list_path(store, universe_path) or []
        parquet_files = [f for f in universe_files if re.fullmatch(r"snapshot_\d{8}_\d{4}\.parquet", str(f.get("name", "")))]
        manifest_files = sorted(
            [f for f in universe_files if re.fullmatch(r"manifest_\d{8}_\d{4}\.json", str(f.get("name", "")))],
            key=lambda x: x["name"],
        )
        per_day_manifest = None
        if manifest_files:
            try:
                payload = json.loads(_read_file_bytes(store, manifest_files[-1]["id"]).decode("utf-8"))
                per_day_manifest = {
                    "name": manifest_files[-1]["name"],
                    "modified_time_utc": manifest_files[-1].get("modifiedTime"),
                    "cycle_minute_ist": payload.get("cycle_minute_ist"),
                    "status": payload.get("status"),
                    "expected_symbol_count": payload.get("expected_symbol_count"),
                    "quoted_symbol_count": payload.get("quoted_symbol_count"),
                    "missing_quote_count": payload.get("missing_quote_count"),
                    "forecast_record_count": payload.get("forecast_record_count"),
                    "matched_outcome_count": payload.get("matched_outcome_count"),
                    "row_count": payload.get("row_count"),
                    "snapshot_drive_path": payload.get("snapshot_drive_path"),
                    "error": str(payload.get("error") or payload.get("manifest_sync_error") or "")[:240] or None,
                }
            except Exception as exc:
                per_day_manifest = {"name": manifest_files[-1]["name"], "read_error": _safe_error(exc)}
        all_universe[day_key] = {
            "snapshot_parquet_files": len(parquet_files),
            "minute_manifest_files": len(manifest_files),
            "latest_manifest": per_day_manifest,
        }

        audit_path = f"production_data/intraday_predictions/outcomes_audit/{y}/{m}/{d}"
        audit_files = _list_path(store, audit_path) or []
        json_reports = sorted(
            [f for f in audit_files if str(f.get("name", "")).endswith(".json") and str(f.get("name", "")).startswith("outcomes_")],
            key=lambda x: (x.get("modifiedTime") or "", x.get("name") or ""),
        )
        audit_summary = {"json_reports": len(json_reports)}
        if json_reports:
            meta = json_reports[-1]
            try:
                report = json.loads(_read_file_bytes(store, meta["id"]).decode("utf-8"))
                audit_summary.update({
                    "latest_report": meta.get("name"),
                    "modified_time_utc": meta.get("modifiedTime"),
                    "status": report.get("status"),
                    "date_range": report.get("date_range"),
                    "ledger_days_found": report.get("ledger_days_found"),
                    "ledger_rows_loaded": report.get("ledger_rows_loaded"),
                    "outcome_rows": report.get("outcome_rows"),
                    "validated_rows": report.get("validated_rows"),
                    "bar_partition_cache_entries": report.get("bar_partition_cache_entries"),
                    "bar_load_errors": len(report.get("bar_load_errors") or []),
                })
            except Exception as exc:
                audit_summary["read_error"] = _safe_error(exc)
        outcomes[day_key] = audit_summary

    worker_status, worker_meta = _read_json(store, "production_data/intraday_predictions/worker_status_latest.json")
    worker = {"exists": worker_status is not None}
    if worker_status is not None:
        worker.update({
            "modified_time_utc": worker_meta.get("modifiedTime") if worker_meta else None,
            "modified_age_minutes": _age_minutes(worker_meta.get("modifiedTime"), now_utc) if worker_meta else None,
            "updated_at_ist": worker_status.get("updated_at_ist"),
            "run_status": worker_status.get("run_status"),
            "cycle_count": worker_status.get("cycle_count"),
            "watchlist_symbol_count": len(worker_status.get("watchlist_symbols") or []),
            "last_cycle_observed_at_ist": (worker_status.get("last_cycle_summary") or {}).get("observed_at_ist"),
            "last_cycle_predictions_written": (worker_status.get("last_cycle_summary") or {}).get("predictions_written"),
            "last_cycle_stock_fetch_errors": (worker_status.get("last_cycle_summary") or {}).get("stock_fetch_errors"),
            "last_cycle_error_count": len((worker_status.get("last_cycle_summary") or {}).get("errors") or []),
        })

    global_manifest = _manifest_summary(
        store,
        "production_data/all_universe_minute_snapshots/latest_manifest.json",
        now_utc,
    )

    today_watch = watch_snapshots[today.isoformat()]
    today_ledger = ledgers[today.isoformat()]
    today_universe = all_universe[today.isoformat()]
    latest_global_age = global_manifest.get("cycle_minute_age_minutes")
    watch_fresh = (
        today_watch.get("file_count_today", 0) > 0
        and (today_watch.get("latest_file_timestamp_age_minutes") is not None)
        and today_watch.get("latest_file_timestamp_age_minutes") <= 5
    )
    watch_actual = today_watch.get("recent_5_snapshot_files_actual_rows", 0) > 0
    watch_predictions = today_watch.get("recent_5_snapshot_files_prediction_rows", 0) > 0
    ledger_recent = today_ledger.get("exists") and (today_ledger.get("modified_age_minutes") is not None) and today_ledger["modified_age_minutes"] <= 10 and int(today_ledger.get("prediction_rows") or 0) > 0
    universe_fresh = (
        today_universe.get("snapshot_parquet_files", 0) > 0
        and today_universe.get("latest_manifest") is not None
        and today_universe["latest_manifest"].get("status") == "OK"
        and _age_minutes(today_universe["latest_manifest"].get("cycle_minute_ist"), now_utc) is not None
        and _age_minutes(today_universe["latest_manifest"].get("cycle_minute_ist"), now_utc) <= 5
    )
    current_actual_matched = int(global_manifest.get("matched_outcome_count") or 0) if global_manifest.get("exists") else 0
    historic_validated = max(
        (int(v.get("validated_rows") or 0) for v in outcomes.values()),
        default=0,
    )

    verdict = {
        "watchlist_minute_snapshots_fresh": bool(watch_fresh),
        "watchlist_actual_observation_rows_present": bool(watch_actual),
        "watchlist_prediction_rows_present_recently": bool(watch_predictions),
        "today_prediction_ledger_recent_and_nonempty": bool(ledger_recent),
        "full_universe_minute_capture_fresh_and_status_ok": bool(universe_fresh),
        "latest_global_manifest_matched_outcome_count": current_actual_matched,
        "latest_historical_outcome_audit_validated_rows": historic_validated,
    }
    verdict["overall"] = (
        "PASS_CURRENT_MINUTE_PREDICTION_AND_ACTUAL_SNAPSHOT_STORAGE"
        if watch_fresh and watch_actual and watch_predictions and ledger_recent and universe_fresh
        else "PARTIAL_OR_NOT_CURRENTLY_CONFIRMED"
    )

    report = {
        "checked_at_ist": now.isoformat(),
        "drive_preflight": drive_safe,
        "current_day": today.isoformat(),
        "lookback_days_scanned": lookback_days,
        "worker_status_latest": worker,
        "watchlist_minute_snapshots_by_day": watch_snapshots,
        "prediction_ledger_by_day": ledgers,
        "watchlist_one_minute_bars_by_day": bars,
        "full_universe_minute_capture_by_day": all_universe,
        "full_universe_latest_manifest": global_manifest,
        "outcome_audits_by_day": outcomes,
        "verdict": verdict,
        "important_interpretation": (
            "A minute snapshot proves a contemporaneous observed price was stored with its forecast. "
            "Matured prediction-versus-target outcomes are separately validated from the prediction ledger "
            "and one-minute bars by the outcome audit. Full-universe quote snapshots are observed quotes, "
            "not a claim that the forecasting model predicts every NSE symbol."
        ),
    }
    print("MINUTE_PERSISTENCE_AUDIT_JSON_BEGIN")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print("MINUTE_PERSISTENCE_AUDIT_JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
