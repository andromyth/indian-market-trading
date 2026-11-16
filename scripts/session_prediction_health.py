#!/usr/bin/env python3
"""Read-only live-session health check for the five-minute NSE pipeline.

Reports the latest canonical quote snapshot, the latest ALL-NSE prediction
manifest, and today's live five-minute OHLCV captures. It never writes to
Drive, never retrains, and never places orders.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import os
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
FOLDER_MIME = "application/vnd.google-apps.folder"
SNAPSHOT_ROOT = "production_data/all_universe_minute_snapshots"
PREDICTION_ROOT = "production_data/all_nse_minute_predictions"
LIVE_HISTORY_ROOT = "production_data/nse_official/intraday_5m_live"


def _list_children(service, parent_id: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    token = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{parent_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken,files(id,name,mimeType,modifiedTime,size)",
                pageSize=1000,
                pageToken=token,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
        files.extend(response.get("files", []))
        token = response.get("nextPageToken")
        if not token:
            return files


def _resolve_folder(store, parts: list[str]) -> dict[str, Any] | None:
    parent = store.root_id
    node = None
    for name in parts:
        node = store._find_child(parent, name, FOLDER_MIME)
        if not node:
            return None
        parent = node["id"]
    return node


def _read_json(service, file_id: str) -> dict[str, Any] | None:
    try:
        raw = service.files().get_media(fileId=file_id).execute()
        return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except Exception:
        return None


def _age_minutes(current: datetime, minute_ist: Any) -> float | None:
    if not minute_ist:
        return None
    try:
        parsed = datetime.fromisoformat(str(minute_ist).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=IST)
        return round((current - parsed.astimezone(IST)).total_seconds() / 60.0, 1)
    except (TypeError, ValueError):
        return None


def _latest_slot(files: list[dict[str, Any]], prefix: str) -> dict[str, Any] | None:
    matches = [f for f in files if str(f.get("name") or "").startswith(prefix)]
    if not matches:
        return None
    return max(matches, key=lambda f: str(f.get("name") or ""))


def _expected_slots(current: datetime) -> int:
    open_dt = current.replace(hour=9, minute=15, second=0, microsecond=0)
    if current < open_dt:
        return 0
    close_dt = current.replace(hour=15, minute=25, second=0, microsecond=0)
    end = min(current, close_dt)
    return int((end - open_dt).total_seconds() // 300) + 1


def snapshot_report(service, store, current: datetime) -> dict[str, Any]:
    day = current.strftime("%Y/%m/%d")
    stamp = current.strftime("%Y%m%d")
    folder = _resolve_folder(store, SNAPSHOT_ROOT.split("/") + [current.strftime("%Y"), current.strftime("%m"), current.strftime("%d")])
    if not folder:
        return {"snapshot_folder_present": False, "snapshot_slot_count": 0}
    files = _list_children(service, folder["id"])
    manifests = [f for f in files if str(f.get("name") or "").startswith(f"manifest_{stamp}_")]
    latest = _latest_slot(manifests, f"manifest_{stamp}_")
    payload = _read_json(service, latest["id"]) if latest else {}
    payload = payload or {}
    expected = _expected_slots(current)
    return {
        "snapshot_folder_present": True,
        "snapshot_slot_count": len(manifests),
        "snapshot_expected_slots": expected,
        "snapshot_missing_slots": max(0, expected - len(manifests)),
        "snapshot_latest_manifest": latest.get("name") if latest else None,
        "snapshot_cycle_minute_ist": payload.get("cycle_minute_ist"),
        "snapshot_age_minutes": _age_minutes(current, payload.get("cycle_minute_ist")),
        "snapshot_status": payload.get("status"),
        "snapshot_assembly_status": payload.get("assembly_status"),
        "snapshot_persisted": payload.get("snapshot_persisted"),
        "snapshot_row_count": payload.get("row_count"),
        "snapshot_quoted_symbol_count": payload.get("quoted_symbol_count"),
        "snapshot_missing_quote_count": payload.get("missing_quote_count"),
        "snapshot_expected_symbol_count": payload.get("expected_symbol_count"),
        "snapshot_worker_count": payload.get("worker_count"),
        "snapshot_shard_count_completed": payload.get("shard_count_completed"),
        "snapshot_shard_count_expected": payload.get("shard_count_expected"),
        "day_folder": day,
    }


def prediction_report(service, store, current: datetime) -> dict[str, Any]:
    stamp = current.strftime("%Y%m%d")
    folder = _resolve_folder(store, PREDICTION_ROOT.split("/") + [current.strftime("%Y"), current.strftime("%m"), current.strftime("%d")])
    if not folder:
        return {"prediction_folder_present": False, "prediction_slot_count": 0}
    files = _list_children(service, folder["id"])
    manifests = [f for f in files if str(f.get("name") or "").startswith(f"manifest_{stamp}_")]
    parquets = [f for f in files if str(f.get("name") or "").startswith(f"predictions_{stamp}_")]
    latest = _latest_slot(manifests, f"manifest_{stamp}_")
    payload = _read_json(service, latest["id"]) if latest else {}
    payload = payload or {}
    return {
        "prediction_folder_present": True,
        "prediction_slot_count": len(manifests),
        "prediction_expected_slots": _expected_slots(current),
        "prediction_missing_slots": max(0, _expected_slots(current) - len(manifests)),
        "prediction_parquet_count": len(parquets),
        "prediction_latest_manifest": latest.get("name") if latest else None,
        "prediction_cycle_minute_ist": payload.get("cycle_minute_ist"),
        "prediction_age_minutes": _age_minutes(current, payload.get("cycle_minute_ist")),
        "prediction_status": payload.get("status"),
        "prediction_count": payload.get("prediction_count"),
        "prediction_eligible_symbol_count": payload.get("eligible_symbol_count"),
        "prediction_snapshot_symbol_count": payload.get("snapshot_symbol_count"),
        "prediction_skipped_missing_quote_count": payload.get("skipped_missing_quote_count"),
        "prediction_actual_quote_count": payload.get("actual_quote_count"),
        "prediction_data_limited_count": payload.get("data_limited_count"),
        "prediction_model_failure_count": payload.get("model_failure_count"),
        "prediction_history_seed_coverage_symbols": payload.get("history_seed_coverage_symbols"),
        "prediction_drive_path": payload.get("prediction_drive_path"),
    }


def live_history_report(service, store, current: datetime) -> dict[str, Any]:
    folder = _resolve_folder(store, LIVE_HISTORY_ROOT.split("/") + [current.strftime("%Y"), current.strftime("%m"), current.strftime("%d")])
    if not folder:
        return {"live_history_folder_present": False, "live_history_capture_count": 0}
    files = _list_children(service, folder["id"])
    captures = [f for f in files if str(f.get("name") or "").startswith("capture_") and str(f.get("name") or "").endswith(".parquet")]
    latest = _latest_slot(captures, "capture_")
    return {
        "live_history_folder_present": True,
        "live_history_capture_count": len(captures),
        "live_history_latest_capture": latest.get("name") if latest else None,
        "live_history_latest_modified_utc": latest.get("modifiedTime") if latest else None,
        "live_history_latest_size": latest.get("size") if latest else None,
    }


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from persistent_store import GoogleDriveStore

    now = datetime.now(IST)
    store = GoogleDriveStore()
    service = store.service
    report: dict[str, Any] = {
        "checked_at_ist": now.isoformat(),
        "trading_day": now.weekday() < 5,
    }
    report.update(snapshot_report(service, store, now))
    report.update(prediction_report(service, store, now))
    report.update(live_history_report(service, store, now))

    snapshot_age = report.get("snapshot_age_minutes")
    prediction_age = report.get("prediction_age_minutes")
    report["snapshot_fresh"] = bool(snapshot_age is not None and snapshot_age <= 12)
    report["prediction_fresh"] = bool(prediction_age is not None and prediction_age <= 15)
    report["prediction_nonempty"] = bool(int(report.get("prediction_count") or 0) > 0)
    report["ohlcv_live_present"] = bool(int(report.get("live_history_capture_count") or 0) > 0)
    verdict = (
        "LIVE_OK"
        if report["snapshot_fresh"] and report["prediction_fresh"] and report["prediction_nonempty"]
        else "LIVE_GAP"
    )
    report["verdict"] = verdict
    print("SESSION_HEALTH_JSON=" + json.dumps(report, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
