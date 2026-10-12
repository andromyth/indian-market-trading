#!/usr/bin/env python3
"""Read-only audit of live-market freshness, active model identity, and realised prediction outcomes.

Run inside the project environment with access to the mounted
Indian_Market_Research Drive. It does not retrain, promote, or trade.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
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
    return {
        "root": str(BULK_ROOT),
        "files_with_parseable_dates": files_seen,
        "distinct_archive_dates": len(ordered),
        "first_archive_date": str(ordered[0]) if ordered else None,
        "latest_archive_date": str(ordered[-1]) if ordered else None,
        "dates_later_than_cash_master": later,
        "master_lags_archives": bool(later),
        "status": "STALE_MASTER" if later else ("NO_ARCHIVES" if not ordered else "NO_LATER_ARCHIVES_FOUND"),
    }


def _latest_pit_readiness() -> dict[str, Any]:
    reports = []
    if PIT_ROOT.exists():
        for path in PIT_ROOT.glob("DELIVERY_PIT_READINESS_*.json"):
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
    return reports[-1] if reports else {"status": "NO_PIT_READINESS_REPORT", "root": str(PIT_ROOT)}


def _delivery_model_manifest() -> dict[str, Any]:
    if not DELIVERY_MANIFEST.exists():
        return {"status": "MANIFEST_NOT_FOUND", "path": str(DELIVERY_MANIFEST)}
    try:
        payload = json.loads(DELIVERY_MANIFEST.read_text(encoding="utf-8"))
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


def main() -> int:
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    master = _cash_master_audit()
    bulk = _bulk_archive_audit(master.get("latest_date"))
    pit = _latest_pit_readiness()
    delivery_manifest = _delivery_model_manifest()
    delivery_outcomes = _realised_delivery_diagnostics()
    intraday_outcomes = _intraday_outcome_coverage()
    intraday = _intraday_model_audit()

    issues = []
    if master.get("status") != "PASS":
        issues.append("Canonical identity-preserved cash master is missing or has identity-integrity problems.")
    if bulk.get("master_lags_archives"):
        issues.append("Official EOD archives contain later dates than the canonical cash research master.")
    if pit.get("snapshot_date") and master.get("latest_date") and pit["snapshot_date"] < master["latest_date"]:
        issues.append("Latest Delivery PIT snapshot predates the canonical cash research master.")
    if delivery_outcomes.get("status") in {
        "NO_ACTIVE_MODEL_DATA", "NO_QUALITY_APPROVED_DATA", "INSUFFICIENT_HISTORY",
        "INSUFFICIENT_SESSION_HISTORY", "AUDIT_FAILED", "NO_DATA",
    }:
        issues.append(
            "The live Delivery accuracy diagnostic has no adequate quality-approved active-model outcome history."
        )
    if intraday.get("status") == "AUDIT_FAILED":
        issues.append("The certified intraday model manifest/target contract could not be verified.")
    if intraday_outcomes.get("intraday_prediction_count", 0) and intraday_outcomes.get("status") == "NO_TARGET_ALIGNED_30M_OUTCOME_VALIDATOR":
        issues.append("Intraday predictions exist, but no target-aligned 30-minute realised-outcome validator was found in the validation ledger.")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_at_ist": now.isoformat(),
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
            "The report is only current when run against the mounted Drive files and persisted prediction/validation ledgers.",
            "Monitoring accuracy is not a sealed out-of-sample result; require later chronological holdout evidence before claiming a target precision.",
            "A clean archive/master date match proves freshness only, not predictive edge or profitability.",
        ],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
