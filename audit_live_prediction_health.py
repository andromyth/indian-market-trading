#!/usr/bin/env python3
"""Read-only audit of live-market freshness, active model identity, and realised prediction outcomes.

Run inside the project environment with access to the mounted
Indian_Market_Research Drive. It does not retrain, promote, or trade.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
NSE_ROOT = BASE / "production_data" / "nse_official"
MASTER = NSE_ROOT / "ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.parquet"
BULK_ROOT = NSE_ROOT / "bulk_eod"
PIT_ROOT = BASE / "features" / "daily"
DELIVERY_MANIFEST = BASE / "models" / "all_nse_delivery_multifactor_replacement_2026" / "ALL_NSE_MULTIFACTOR_59_MODEL_MANIFEST_2026.json"
REPORT = BASE / "diagnostics" / "live_prediction_health" / "live_prediction_health_latest.json"
DATE_RE = re.compile(r"_(\d{8})_F_0000\.(?:parquet|csv\.zip)$", re.IGNORECASE)
PIT_DATE_RE = re.compile(r"DELIVERY_PIT_(?:FEATURES|READINESS)_(\d{4}-\d{2}-\d{2})\.(?:parquet|json)$")


def _date_from_file(path: Path):
    match = DATE_RE.search(path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _cash_master_audit() -> dict[str, Any]:
    if not MASTER.exists() or MASTER.stat().st_size == 0:
        return {"status": "MISSING_MASTER", "path": str(MASTER)}
    con = duckdb.connect(database=":memory:")
    try:
        m = str(MASTER).replace("'", "''")
        r = con.execute(f"""
            SELECT COUNT(*), COUNT(DISTINCT row_identity_key),
              COUNT(*) FILTER (WHERE row_identity_key IS NULL),
              COUNT(DISTINCT symbol), COUNT(DISTINCT TRY_CAST(date AS DATE)),
              MIN(TRY_CAST(date AS DATE)), MAX(TRY_CAST(date AS DATE))
            FROM read_parquet('{m}')
        """).fetchone()
        rows, keys, null_keys = int(r[0] or 0), int(r[1] or 0), int(r[2] or 0)
        return {
            "status": "PASS" if null_keys == 0 and rows == keys else "IDENTITY_INTEGRITY_WARNING",
            "path": str(MASTER),
            "rows": rows,
            "distinct_identity_keys": keys,
            "duplicate_identity_key_rows": rows - keys - null_keys,
            "null_identity_key_rows": null_keys,
            "symbols": int(r[3] or 0),
            "trading_dates": int(r[4] or 0),
            "first_date": str(r[5]) if r[5] else None,
            "latest_date": str(r[6]) if r[6] else None,
        }
    finally:
        con.close()


def _expected_missing_sessions(master_latest: str | None, available_dates: set[date]) -> tuple[list[str], int]:
    if not master_latest or not available_dates:
        return [], 0
    latest_archive = max(available_dates)
    start = date.fromisoformat(master_latest) + timedelta(days=1)
    if start > latest_archive:
        return [], 0
    try:
        from nse_bulk_eod_ingest import business_days
        expected = list(business_days(start, latest_archive))
    except Exception:
        expected = [
            day.date()
            for day in pd.date_range(start, latest_archive, freq="B")
        ]
    return [d.isoformat() for d in expected if d not in available_dates], len(expected)


def _bulk_archive_audit(master_latest: str | None) -> dict[str, Any]:
    dates = set()
    files_seen = 0
    for pattern in ("BhavCopy_NSE_CM_0_0_0_*_F_0000.parquet", "BhavCopy_NSE_CM_0_0_0_*_F_0000.csv.zip"):
        for path in (BULK_ROOT.rglob(pattern) if BULK_ROOT.exists() else []):
            day = _date_from_file(path)
            if day is not None:
                files_seen += 1
                dates.add(day)
    ordered = sorted(dates)
    later = [str(d) for d in ordered if master_latest and str(d) > master_latest]
    missing_sessions, expected_sessions_checked = _expected_missing_sessions(master_latest, dates)
    return {
        "root": str(BULK_ROOT),
        "files_with_parseable_dates": files_seen,
        "distinct_archive_dates": len(ordered),
        "first_archive_date": str(ordered[0]) if ordered else None,
        "latest_archive_date": str(ordered[-1]) if ordered else None,
        "dates_later_than_cash_master": later,
        "master_lags_archives": bool(later),
        "expected_sessions_checked_after_master_date": expected_sessions_checked,
        "missing_expected_archive_dates": missing_sessions,
        "status": "ARCHIVE_GAP" if missing_sessions else ("STALE_MASTER" if later else ("NO_ARCHIVES" if not ordered else "NO_LATER_ARCHIVES_FOUND")),
    }


def _latest_pit_readiness(root: Path = PIT_ROOT) -> dict[str, Any]:
    reports = []
    if root.exists():
        for path in root.glob("DELIVERY_PIT_READINESS_*.json"):
            match = PIT_DATE_RE.fullmatch(path.name)
            if not match:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                reports.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            reports.append({
                "path": str(path),
                "filename_date": match.group(1),
                "snapshot_date": payload.get("snapshot_date"),
                "generated_at_utc": payload.get("generated_at_utc"),
                "rows": payload.get("rows"),
                "symbols": payload.get("symbols"),
                "feature_count": payload.get("feature_count"),
                "source_status": payload.get("source_status"),
                "group_readiness": payload.get("group_readiness"),
                "production_feature_coverage_pct": payload.get("production_feature_coverage_pct"),
                "production_rows_all_59_features": payload.get("production_rows_all_59_features"),
                "production_data_quality_gate_pct": payload.get("production_data_quality_gate_pct"),
            })
    reports.sort(key=lambda item: (item.get("snapshot_date") or item.get("filename_date") or ""))
    return reports[-1] if reports else {"status": "NO_PIT_READINESS_REPORT", "root": str(root)}


def _delivery_model_manifest(path: Path = DELIVERY_MANIFEST) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MANIFEST_NOT_FOUND", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "MANIFEST_INVALID", "error": f"{type(exc).__name__}: {exc}"}
    artifacts = payload.get("artifacts", {}) or {}
    horizons = {}
    for horizon, item in artifacts.items():
        metrics = item.get("metrics", {}) if isinstance(item, dict) else {}
        horizons[str(horizon)] = {
            "status": item.get("status") if isinstance(item, dict) else None,
            "metrics": metrics,
            "classifier_sha256": item.get("classifier_sha256") if isinstance(item, dict) else None,
            "regressor_sha256": item.get("regressor_sha256") if isinstance(item, dict) else None,
        }
    return {
        "status": payload.get("promotion_status", "UNKNOWN"),
        "engine_version": payload.get("engine_version"),
        "universe": payload.get("universe"),
        "trainable_symbol_count": payload.get("trainable_symbol_count"),
        "feature_count": payload.get("feature_count"),
        "panel_input_signature": payload.get("panel_input_signature"),
        "artifacts": horizons,
        "warning": "Training holdout metrics in a candidate manifest are not a substitute for the live completed-outcome ledger and a sealed chronological holdout.",
    }


def _realised_delivery_diagnostics() -> dict[str, Any]:
    try:
        from prediction_ledger import load_records, load_validation_records
        from delivery_validation_diagnostics import build_report
        preds = pd.DataFrame(load_records())
        vals = pd.DataFrame(load_validation_records())
        return build_report(preds, vals)
    except Exception as exc:
        return {
            "status": "AUDIT_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "note": "No accuracy figure is inferred when prediction/validation ledgers cannot be read.",
        }


def _intraday_outcome_coverage() -> dict[str, Any]:
    """Distinguish intraday prediction logs from true 30-minute realised outcomes."""
    try:
        from prediction_ledger import load_records, load_validation_records
        preds = pd.DataFrame(load_records())
        vals = pd.DataFrame(load_validation_records())
        if preds.empty:
            return {"status": "NO_INTRADAY_PREDICTIONS", "prediction_count": 0}
        mode = preds.get("mode", pd.Series("", index=preds.index)).astype(str).str.lower()
        horizon = preds.get("prediction_horizon", pd.Series("", index=preds.index)).astype(str).str.lower()
        intraday = preds.loc[mode.eq("intraday") | horizon.isin({"30m", "30min", "30_minutes"})].copy()
        if intraday.empty:
            return {"status": "NO_INTRADAY_PREDICTIONS", "prediction_count": 0}
        fields = {
            "realised_return": ["actual_return_30m", "actual_30m_return", "realized_return_30m", "realised_return_30m"],
            "realised_direction": ["actual_direction_30m", "actual_30m_direction", "realized_direction_30m", "realised_direction_30m"],
            "validation_status": ["validation_status"],
        }
        available = {kind: [name for name in names if name in vals.columns] for kind, names in fields.items()}
        ids = set(intraday.get("prediction_id", pd.Series(dtype=str)).dropna().astype(str))
        matched = vals.loc[vals.get("prediction_id", pd.Series(index=vals.index, dtype=str)).astype(str).isin(ids)].copy() if not vals.empty and "prediction_id" in vals.columns else pd.DataFrame()
        target_outcome_available = bool(available["realised_return"] or available["realised_direction"])
        validated_count = 0
        if not matched.empty and "validation_status" in matched.columns:
            validated_count = int(matched["validation_status"].astype(str).str.upper().eq("VALIDATED").sum())
        return {
            "status": "TARGET_ALIGNED_OUTCOMES_AVAILABLE" if target_outcome_available else "NO_TARGET_ALIGNED_30M_OUTCOME_VALIDATOR",
            "intraday_prediction_count": int(len(intraday)),
            "prediction_ids_available": int(len(ids)),
            "validation_rows_matched_by_prediction_id": int(len(matched)),
            "matched_validated_rows": validated_count,
            "realised_30m_fields_found_in_validation_ledger": available["realised_return"] + available["realised_direction"],
            "validation_ledger_fields": list(vals.columns) if not vals.empty else [],
            "accuracy_claim_allowed": False,
            "note": (
                "A 30-minute intraday score requires future 30-minute realised price outcomes matched to the exact prediction timestamp. "
                "Delivery D+1 validation rows are not valid substitutes."
            ),
        }
    except Exception as exc:
        return {
            "status": "AUDIT_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "accuracy_claim_allowed": False,
        }


def _intraday_model_audit() -> dict[str, Any]:
    try:
        from production_model import model_status, model_target_contract
        return {"model_status": model_status(), "target_contract": model_target_contract()}
    except Exception as exc:
        return {"status": "AUDIT_FAILED", "error": f"{type(exc).__name__}: {exc}"}





REMOTE_CASH_META = "production_data/nse_official/ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.meta.json"
REMOTE_CASH_MASTER = "production_data/nse_official/ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.parquet"
REMOTE_DELIVERY_MANIFEST = "models/all_nse_delivery_multifactor_replacement_2026/ALL_NSE_MULTIFACTOR_59_MODEL_MANIFEST_2026.json"
REMOTE_PIT_FOLDER = "features/daily"
REMOTE_HEALTH_REPORT = "diagnostics/live_prediction_health/live_prediction_health_latest.json"


def _drive_file_id(store, remote_path: str, *, folder: bool = False) -> str | None:
    parts = [part for part in remote_path.replace("\\\\", "/").split("/") if part]
    if not parts:
        return None
    parent = store.root_id
    folder_type = "application/vnd.google-apps.folder"
    for part in parts[:-1]:
        hit = store._find_child(parent, part, folder_type)
        if not hit:
            return None
        parent = hit["id"]
    hit = store._find_child(parent, parts[-1], folder_type if folder else None)
    return hit.get("id") if hit else None


def _drive_download(store, remote_path: str, destination: Path) -> bool:
    file_id = _drive_file_id(store, remote_path)
    if not file_id:
        return False
    from googleapiclient.http import MediaIoBaseDownload
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".download.tmp")
    with temp.open("wb") as handle:
        downloader = MediaIoBaseDownload(
            handle, store.service.files().get_media(fileId=file_id)
        )
        done = False
        while not done:
            _, done = downloader.next_chunk()
    temp.replace(destination)
    return destination.exists() and destination.stat().st_size > 0


def _drive_list(store, query: str) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    token = None
    while True:
        params = {
            "q": query,
            "spaces": "drive",
            "fields": "nextPageToken,files(id,name,mimeType,modifiedTime,size,parents)",
            "pageSize": 1000,
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
        }
        if token:
            params["pageToken"] = token
        response = store.service.files().list(**params).execute()
        files.extend(response.get("files", []))
        token = response.get("nextPageToken")
        if not token:
            break
    return files


def _drive_json_by_id(store, file_id: str, work_root: Path, name: str) -> tuple[dict[str, Any], Path | None]:
    from googleapiclient.http import MediaIoBaseDownload
    destination = work_root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        downloader = MediaIoBaseDownload(
            handle, store.service.files().get_media(fileId=file_id)
        )
        done = False
        while not done:
            _, done = downloader.next_chunk()
    try:
        return json.loads(destination.read_text(encoding="utf-8")), destination
    except Exception:
        return {}, destination


def _drive_cash_master_audit(store, work_root: Path) -> dict[str, Any]:
    file_id = _drive_file_id(store, REMOTE_CASH_MASTER)
    if not file_id:
        return {"status": "MISSING_MASTER", "remote_path": REMOTE_CASH_MASTER}
    meta_id = _drive_file_id(store, REMOTE_CASH_META)
    meta: dict[str, Any] = {}
    if meta_id:
        meta, _ = _drive_json_by_id(store, meta_id, work_root, "master-meta.json")
    file_meta = store.service.files().get(
        fileId=file_id, fields="id,name,modifiedTime,size"
    ).execute()
    duplicate_rows = meta.get(
        "duplicate_identity_key_rows",
        meta.get("duplicate_row_identity_keys"),
    )
    latest = meta.get("max_date")
    trustworthy = latest is not None and duplicate_rows is not None
    return {
        "status": (
            "PASS" if trustworthy and int(duplicate_rows or 0) == 0
            else "MASTER_METADATA_STALE_OR_UNAVAILABLE"
        ),
        "remote_path": REMOTE_CASH_MASTER,
        "metadata_path": REMOTE_CASH_META if meta_id else None,
        "metadata_based_audit": True,
        "full_master_downloaded_for_audit": False,
        "rows": meta.get("rows"),
        "symbols": meta.get("symbols"),
        "trading_dates": meta.get("dates"),
        "first_date": meta.get("min_date"),
        "latest_date": latest,
        "duplicate_identity_key_rows": duplicate_rows,
        "last_incremental_refresh_utc": meta.get("last_incremental_refresh_utc"),
        "master_modified_time": file_meta.get("modifiedTime"),
        "master_size_bytes": int(file_meta.get("size") or 0),
    }


def _drive_bulk_archive_audit(store, master_latest: str | None) -> dict[str, Any]:
    files = _drive_list(
        store,
        "name contains 'BhavCopy_NSE_CM_0_0_0_' and trashed = false",
    )
    dates: set[date] = set()
    file_count = 0
    for item in files:
        name = str(item.get("name", ""))
        if not (name.lower().endswith(".parquet") or name.lower().endswith(".csv.zip")):
            continue
        day = _date_from_file(Path(name))
        if day is not None:
            dates.add(day)
            file_count += 1
    ordered = sorted(dates)
    later = [str(d) for d in ordered if master_latest and str(d) > master_latest]
    missing_sessions, expected_sessions_checked = _expected_missing_sessions(master_latest, dates)
    return {
        "source": "Google Drive file listing",
        "files_with_parseable_dates": file_count,
        "distinct_archive_dates": len(ordered),
        "first_archive_date": str(ordered[0]) if ordered else None,
        "latest_archive_date": str(ordered[-1]) if ordered else None,
        "dates_later_than_cash_master": later,
        "master_lags_archives": bool(later),
        "expected_sessions_checked_after_master_date": expected_sessions_checked,
        "missing_expected_archive_dates": missing_sessions,
        "status": "ARCHIVE_GAP" if missing_sessions else ("STALE_MASTER" if later else ("NO_ARCHIVES" if not ordered else "NO_LATER_ARCHIVES_FOUND")),
    }


def _prepare_drive_audit_inputs(store, work_root: Path) -> tuple[dict[str, Any], Path, Path]:
    """Download only small readiness/model manifests; never fetch the full cash master."""
    pit_folder_id = _drive_file_id(store, REMOTE_PIT_FOLDER, folder=True)
    readiness: dict[str, Any] = {"status": "NO_PIT_READINESS_REPORT", "root": REMOTE_PIT_FOLDER}
    readiness_root = work_root / "features" / "daily"
    readiness_root.mkdir(parents=True, exist_ok=True)
    if pit_folder_id:
        candidates = store.service.files().list(
            q=f"'{pit_folder_id}' in parents and name contains 'DELIVERY_PIT_READINESS_' and trashed = false",
            spaces="drive",
            fields="files(id,name,modifiedTime,size)",
            pageSize=1000,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute().get("files", [])
        candidates = sorted(
            (x for x in candidates if PIT_DATE_RE.fullmatch(str(x.get("name", "")))),
            key=lambda x: str(x["name"]),
        )
        if candidates:
            chosen = candidates[-1]
            _drive_json_by_id(store, chosen["id"], work_root, f"features/daily/{chosen['name']}")
    manifest_path = work_root / REMOTE_DELIVERY_MANIFEST
    _drive_download(store, REMOTE_DELIVERY_MANIFEST, manifest_path)
    return readiness, readiness_root, manifest_path
\n\ndef main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive-api",
        action="store_true",
        help="Audit current Google Drive artifacts through the Drive API without downloading the full cash master.",
    )
    args = parser.parse_args()
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    drive_store = None
    work_root = None
    temp_context = None

    if args.drive_api:
        from persistent_store import GoogleDriveStore
        drive_store = GoogleDriveStore()
        temp_context = tempfile.TemporaryDirectory(prefix="live-prediction-health-")
        work_root = Path(temp_context.name)
        master = _drive_cash_master_audit(drive_store, work_root)
        bulk = _drive_bulk_archive_audit(drive_store, master.get("latest_date"))
        _, pit_root, manifest_path = _prepare_drive_audit_inputs(drive_store, work_root)
        pit = _latest_pit_readiness(pit_root)
        delivery_manifest = _delivery_model_manifest(manifest_path)
        report_path = work_root / "live_prediction_health_latest.json"
    else:
        master = _cash_master_audit()
        bulk = _bulk_archive_audit(master.get("latest_date"))
        pit = _latest_pit_readiness()
        delivery_manifest = _delivery_model_manifest()
        report_path = REPORT

    delivery_outcomes = _realised_delivery_diagnostics()
    intraday_outcomes = _intraday_outcome_coverage()
    intraday = _intraday_model_audit()

    issues = []
    if master.get("status") != "PASS":
        issues.append("Canonical identity-preserved cash master is missing or its integrity metadata is unavailable/unapproved.")
    if bulk.get("master_lags_archives"):
        issues.append("Official EOD archives contain later dates than the canonical cash research master.")
    if bulk.get("missing_expected_archive_dates"):
        issues.append("One or more expected NSE trading-session archives are missing after the canonical cash master date.")
    if pit.get("snapshot_date") and master.get("latest_date") and pit["snapshot_date"] < master["latest_date"]:
        issues.append("Latest Delivery PIT snapshot predates the canonical cash research master.")
    if pit.get("status") == "NO_PIT_READINESS_REPORT":
        issues.append("No Delivery PIT readiness report could be found.")
    if delivery_outcomes.get("status") in {
        "NO_ACTIVE_MODEL_DATA", "NO_QUALITY_APPROVED_DATA", "INSUFFICIENT_HISTORY",
        "INSUFFICIENT_SESSION_HISTORY", "AUDIT_FAILED", "NO_DATA",
    }:
        issues.append("The live Delivery accuracy diagnostic has no adequate quality-approved active-model outcome history.")
    if intraday.get("status") == "AUDIT_FAILED":
        issues.append("The certified intraday model manifest/target contract could not be verified.")
    if intraday_outcomes.get("intraday_prediction_count", 0) and intraday_outcomes.get("status") == "NO_TARGET_ALIGNED_30M_OUTCOME_VALIDATOR":
        issues.append("Intraday predictions exist, but no target-aligned 30-minute realised-outcome validator was found in the validation ledger.")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_at_ist": now.isoformat(),
        "audit_source": "Google Drive API" if args.drive_api else "mounted Google Drive filesystem",
        "read_only": True,
        "model_weights_changed": False,
        "model_promotion_performed": False,
        "orders_placed": False,
        "canonical_cash_master": master,
        "official_eod_archive_coverage": bulk,
        "latest_delivery_pit_readiness": pit,
        "delivery_model_manifest": delivery_manifest,
        "delivery_realised_outcome_diagnostics": delivery_outcomes,
        "intraday_model_audit": intraday,
        "intraday_realised_outcome_coverage": intraday_outcomes,
        "issues": issues,
        "overall_status": "NEEDS_REPAIR_OR_MORE_VALIDATION" if issues else "NO_BLOCKER_FOUND_BY_STATIC_AUDIT",
        "limitations": [
            "In Drive API mode, canonical master date/identity counts are taken from its write-validated metadata; the large master Parquet is not downloaded.",
            "The report is only current when run against the live Drive files and persisted prediction/validation ledgers.",
            "Monitoring accuracy is not a sealed out-of-sample result; require later chronological holdout evidence before claiming a target precision.",
            "A clean archive/master date match proves freshness only, not predictive edge or profitability.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if drive_store is not None:
        drive_store.upload_or_update(
            report_path,
            REMOTE_HEALTH_REPORT,
            mime_type="application/json",
        )
        print(json.dumps(report, indent=2, default=str))
        if temp_context is not None:
            temp_context.cleanup()
    else:
        print(json.dumps(report, indent=2, default=str))
    return 0 if not issues else 2

if __name__ == "__main__":
    raise SystemExit(main())
