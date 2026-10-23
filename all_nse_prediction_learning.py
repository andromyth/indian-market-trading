#!/usr/bin/env python3
"""Quality-gated outcome learning for the deployed full-NSE 30-minute predictor.

Partial capture files are retained in Drive for operations/audit, but only a slot
whose quote and prediction manifests are complete may contribute to calibration.
This module changes only the calibrated probability/return/signal overlay; it does
not replace classifier/regressor joblib weights or place orders.
"""
from __future__ import annotations

import argparse
import io
import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from audit_all_universe_minute_snapshots import (
    _download_json,
    _find_folder,
    _list_children,
)
from prediction_learning_engine import _run_calibration_autopilot
from persistent_store import GoogleDriveStore

IST = ZoneInfo("Asia/Kolkata")
PREDICTION_ROOT = "production_data/all_nse_minute_predictions"
QUOTE_ROOT = "production_data/all_universe_minute_snapshots"
LEARNING_ROOT = f"{PREDICTION_ROOT}/learning"
MIME_FOLDER = "application/vnd.google-apps.folder"
MAX_TARGET_QUOTE_DISTANCE_SECONDS = 180
MODEL_VERSION_PREFIX = "all_nse_intraday__"


def _number(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _true(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def prediction_manifest_is_clean(manifest: dict[str, Any]) -> bool:
    """Require the full-NSE inference slot to be complete and model-ready."""
    try:
        expected = int(manifest.get("expected_symbol_count") or 0)
        rows = int(manifest.get("row_count") or 0)
        predictions = int(manifest.get("prediction_count") or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        manifest.get("status") == "OK"
        and manifest.get("quality_gate_pass") is True
        and expected > 0
        and rows == expected
        and predictions == expected
        and int(manifest.get("missing_quote_count") or 0) == 0
        and int(manifest.get("data_limited_count") or 0) == 0
        and int(manifest.get("untrained_symbol_count") or 0) == 0
        and int(manifest.get("model_failure_count") or 0) == 0
        and manifest.get("trained_symbol_manifest_available") is True
        and manifest.get("history_ready") is True
        and int(manifest.get("history_seed_coverage_symbols") or 0) == expected
        and int(manifest.get("history_seed_failed_batches") or 0) == 0
    )


def quote_manifest_is_clean(manifest: dict[str, Any]) -> bool:
    """Require every official-universe quote row and every provider batch."""
    try:
        expected = int(manifest.get("expected_symbol_count") or 0)
        rows = int(manifest.get("row_count") or 0)
        quoted = int(manifest.get("quoted_symbol_count") or 0)
    except (TypeError, ValueError):
        return False
    pending_snapshot_errors = len((manifest.get("pending_snapshot_recovery") or {}).get("errors") or [])
    pending_manifest_errors = len((manifest.get("pending_manifest_recovery") or {}).get("errors") or [])
    return bool(
        manifest.get("status") == "OK"
        and expected > 0
        and rows == expected
        and quoted == expected
        and int(manifest.get("missing_quote_count") or 0) == 0
        and int(manifest.get("unmapped_symbol_count") or 0) == 0
        and pending_snapshot_errors == 0
        and pending_manifest_errors == 0
    )


def match_matured_outcomes(
    predictions: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    max_target_distance_seconds: int = MAX_TARGET_QUOTE_DISTANCE_SECONDS,
) -> pd.DataFrame:
    """Join forecasts to nearest clean quote close near the 30-minute target.

    Quote timestamps are provider observation times, not exchange-timestamped
    candles. The nearest same-symbol quote must be within three minutes of the
    target; otherwise the outcome remains unvalidated and is not used for fit.
    """
    if predictions.empty:
        return pd.DataFrame()
    quote_frame = quotes.copy()
    if not quote_frame.empty:
        quote_frame["symbol"] = quote_frame["symbol"].astype(str).str.upper().str.replace(".NS", "", regex=False)
        quote_frame["_quote_ts"] = pd.to_datetime(quote_frame["quote_observed_at_utc"], errors="coerce", utc=True)
        quote_frame["_quote_price"] = pd.to_numeric(quote_frame["actual_observed_price"], errors="coerce")
        quote_frame = quote_frame.loc[
            quote_frame["_quote_ts"].notna()
            & quote_frame["_quote_price"].gt(0)
            & quote_frame["quote_status"].astype(str).str.upper().eq("PASS")
        ].copy()
        quote_frame = quote_frame.sort_values("_quote_ts")
    lookup: dict[str, pd.DataFrame] = {
        symbol: group for symbol, group in quote_frame.groupby("symbol", sort=False)
    } if not quote_frame.empty else {}

    out: list[dict[str, Any]] = []
    for row in predictions.to_dict(orient="records"):
        symbol = str(row.get("symbol") or "").strip().upper().replace(".NS", "")
        target = pd.to_datetime(row.get("forecast_target_timestamp_utc"), errors="coerce", utc=True)
        decision_ts = pd.to_datetime(row.get("actual_observed_at_utc"), errors="coerce", utc=True)
        ref_price = _number(row.get("actual_market_value"))
        raw_probability = _number(row.get("probability_up_30m_raw", row.get("probability_up_30m")))
        raw_return = _number(row.get("expected_return_30m_raw", row.get("expected_return_30m")))
        version = str(row.get("model_version") or "LEGACY_UNKNOWN")
        status = "PENDING_TARGET_QUOTE"
        actual_price = float("nan")
        actual_quote_ts = pd.NaT
        actual_return = float("nan")
        quote_group = lookup.get(symbol)
        if (
            quote_group is not None and not pd.isna(target) and not pd.isna(decision_ts)
            and np.isfinite(ref_price) and ref_price > 0
        ):
            distances = (quote_group["_quote_ts"] - target).abs().dt.total_seconds()
            idx = distances.idxmin()
            distance = float(distances.loc[idx])
            if distance <= max_target_distance_seconds:
                actual_quote_ts = quote_group.loc[idx, "_quote_ts"]
                actual_price = float(quote_group.loc[idx, "_quote_price"])
                actual_return = actual_price / ref_price - 1.0
                status = "VALIDATED"
        out.append({
            "forecast_key": str(row.get("forecast_key") or ""),
            "symbol": symbol,
            "decision_timestamp_utc": decision_ts.isoformat() if not pd.isna(decision_ts) else None,
            "target_timestamp_utc": target.isoformat() if not pd.isna(target) else None,
            "actual_quote_timestamp_utc": actual_quote_ts.isoformat() if not pd.isna(actual_quote_ts) else None,
            "target_quote_distance_seconds": (
                abs((actual_quote_ts - target).total_seconds())
                if not pd.isna(actual_quote_ts) and not pd.isna(target) else None
            ),
            "reference_price": ref_price if np.isfinite(ref_price) else None,
            "actual_target_price": actual_price if np.isfinite(actual_price) else None,
            "actual_return": actual_return if np.isfinite(actual_return) else None,
            "probability_up": raw_probability if np.isfinite(raw_probability) else None,
            "probability_up_calibrated": _number(row.get("probability_up_30m_calibrated")),
            "expected_return": raw_return if np.isfinite(raw_return) else None,
            "expected_return_calibrated": _number(row.get("expected_return_30m_calibrated")),
            "signal": str(row.get("signal_raw") or row.get("raw_model_signal") or row.get("signal") or "HOLD").upper(),
            "signal_calibrated": row.get("signal_calibrated"),
            "calibration_profile_id": row.get("calibration_profile_id"),
            "model_version": version,
            "model_family": str(row.get("model_family") or "UNKNOWN"),
            "horizon_min": 30,
            "input_age_seconds": _number(row.get("actual_quote_age_seconds")),
            "feature_coverage": _number(row.get("data_readiness_score")),
            "validation_status": status,
            "source": "all_nse_intraday_production",
            "quote_basis": "nearest provider quote observation within 180s of nominal target; not exchange-stamped 1m OHLCV",
        })
    return pd.DataFrame(out)


def _day_folder_files(store: Any, root: str, day: date) -> list[dict[str, Any]]:
    folder_id = _find_folder(store, root.split("/") + [f"{day:%Y}", f"{day:%m}", f"{day:%d}"])
    return _list_children(store, folder_id) if folder_id else []


def _download_bytes(store: Any, file_id: str) -> bytes:
    payload = store.service.files().get_media(fileId=file_id).execute()
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if hasattr(payload, "getvalue"):
        return payload.getvalue()
    return str(payload).encode("utf-8")


def _file_map(files: list[dict[str, Any]], prefix: str, suffix: str) -> dict[str, dict[str, Any]]:
    result = {}
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{8}}_\d{{4}}){re.escape(suffix)}$")
    for item in files:
        match = pattern.match(str(item.get("name") or ""))
        if match:
            result[match.group(1)] = item
    return result


def load_clean_session_data(
    store: Any, start_date: date, end_date: date
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prediction_rows: list[dict[str, Any]] = []
    quote_rows: list[dict[str, Any]] = []
    report = {
        "calendar_days_scanned": (end_date - start_date).days + 1,
        "prediction_slots_seen": 0,
        "prediction_slots_eligible": 0,
        "prediction_slots_quarantined": 0,
        "quote_slots_seen": 0,
        "quote_slots_eligible": 0,
        "quote_slots_quarantined": 0,
        "prediction_rows_retained_for_diagnostics": 0,
        "clean_prediction_rows": 0,
        "quote_rows_retained": 0,
        "manifest_errors": [],
        "parquet_errors": [],
        "excluded_reasons": {},
    }

    current = start_date
    while current <= end_date:
        pred_files = _day_folder_files(store, PREDICTION_ROOT, current)
        quote_files = _day_folder_files(store, QUOTE_ROOT, current)
        pred_manifests = _file_map(pred_files, "manifest_", ".json")
        pred_parquets = _file_map(pred_files, "predictions_", ".parquet")
        quote_manifests = _file_map(quote_files, "manifest_", ".json")
        quote_parquets = _file_map(quote_files, "snapshot_", ".parquet")
        clean_quote_keys: set[str] = set()

        for key, mf in quote_manifests.items():
            report["quote_slots_seen"] += 1
            try:
                manifest = _download_json(store, mf["id"])
            except Exception as exc:
                report["manifest_errors"].append(f"quote/{current}/{key}: {type(exc).__name__}: {exc}")
                report["quote_slots_quarantined"] += 1
                report["excluded_reasons"]["quote_manifest_unreadable"] = report["excluded_reasons"].get("quote_manifest_unreadable", 0) + 1
                continue
            if not quote_manifest_is_clean(manifest) or key not in quote_parquets:
                report["quote_slots_quarantined"] += 1
                reason = "quote_incomplete_or_snapshot_missing"
                report["excluded_reasons"][reason] = report["excluded_reasons"].get(reason, 0) + 1
                continue
            try:
                raw = _download_bytes(store, quote_parquets[key]["id"])
                frame = pd.read_parquet(io.BytesIO(raw))
                expected = int(manifest.get("expected_symbol_count") or 0)
                if len(frame) != expected or frame["symbol"].astype(str).str.upper().nunique() != expected:
                    raise ValueError(f"quote row/universe mismatch: rows={len(frame)} unique={frame['symbol'].nunique()} expected={expected}")
                frame["slot_key_ist"] = key
                quote_rows.extend(frame.to_dict(orient="records"))
                clean_quote_keys.add(key)
                report["quote_slots_eligible"] += 1
            except Exception as exc:
                report["parquet_errors"].append(f"quote/{current}/{key}: {type(exc).__name__}: {exc}")
                report["quote_slots_quarantined"] += 1
                report["excluded_reasons"]["quote_parquet_invalid"] = report["excluded_reasons"].get("quote_parquet_invalid", 0) + 1

        for key, mf in pred_manifests.items():
            report["prediction_slots_seen"] += 1
            try:
                manifest = _download_json(store, mf["id"])
            except Exception as exc:
                report["manifest_errors"].append(f"prediction/{current}/{key}: {type(exc).__name__}: {exc}")
                report["prediction_slots_quarantined"] += 1
                report["excluded_reasons"]["prediction_manifest_unreadable"] = report["excluded_reasons"].get("prediction_manifest_unreadable", 0) + 1
                continue
            # Prediction rows are eligible only if both the prediction model and
            # the contemporaneous all-universe quote capture passed their full gates.
            if not prediction_manifest_is_clean(manifest) or key not in pred_parquets or key not in clean_quote_keys:
                report["prediction_slots_quarantined"] += 1
                reason = "prediction_incomplete_or_quote_slot_not_clean"
                report["excluded_reasons"][reason] = report["excluded_reasons"].get(reason, 0) + 1
                continue
            try:
                raw = _download_bytes(store, pred_parquets[key]["id"])
                frame = pd.read_parquet(io.BytesIO(raw))
                expected = int(manifest.get("expected_symbol_count") or 0)
                if len(frame) != expected or frame["symbol"].astype(str).str.upper().nunique() != expected:
                    raise ValueError(f"prediction row/universe mismatch: rows={len(frame)} unique={frame['symbol'].nunique()} expected={expected}")
                report["prediction_rows_retained_for_diagnostics"] += len(frame)
                symbol = frame["symbol"].astype(str).str.upper().str.replace(".NS", "", regex=False)
                probability = pd.to_numeric(frame.get("probability_up_30m"), errors="coerce")
                quote_age = pd.to_numeric(frame.get("actual_quote_age_seconds"), errors="coerce")
                valid_row = (
                    frame.get("prediction_status", pd.Series("", index=frame.index)).astype(str).eq("PASS")
                    & frame.get("actual_quote_status", pd.Series("", index=frame.index)).astype(str).str.upper().eq("PASS")
                    & frame.get("trained_for_symbol", pd.Series(False, index=frame.index)).map(_true)
                    & ~frame.get("data_limited", pd.Series(True, index=frame.index)).map(_true)
                    & probability.between(0, 1)
                    & quote_age.le(120)
                    & pd.to_numeric(frame.get("actual_market_value"), errors="coerce").gt(0)
                    & frame.get("forecast_target_timestamp_utc", pd.Series(None, index=frame.index)).notna()
                )
                clean = frame.loc[valid_row].copy()
                clean["symbol"] = symbol.loc[valid_row]
                if "model_version" not in clean:
                    clean["model_version"] = str(manifest.get("model_version") or "LEGACY_UNKNOWN")
                else:
                    clean["model_version"] = clean["model_version"].fillna(
                        str(manifest.get("model_version") or "LEGACY_UNKNOWN")
                    ).astype(str)
                clean["slot_key_ist"] = key
                prediction_rows.extend(clean.to_dict(orient="records"))
                report["clean_prediction_rows"] += len(clean)
                report["prediction_slots_eligible"] += 1
                if len(clean) != expected:
                    # The whole slot remains auditable; only individually clean
                    # rows become candidates, and an unexpected shortfall is flagged.
                    report["excluded_reasons"]["row_level_quality_failures_in_manifest_ok_slot"] = report["excluded_reasons"].get("row_level_quality_failures_in_manifest_ok_slot", 0) + (expected - len(clean))
            except Exception as exc:
                report["parquet_errors"].append(f"prediction/{current}/{key}: {type(exc).__name__}: {exc}")
                report["prediction_slots_quarantined"] += 1
                report["excluded_reasons"]["prediction_parquet_invalid"] = report["excluded_reasons"].get("prediction_parquet_invalid", 0) + 1
        current += timedelta(days=1)

    predictions = pd.DataFrame(prediction_rows)
    quotes = pd.DataFrame(quote_rows)
    report["quote_rows_retained"] = len(quotes)
    if not predictions.empty:
        decision_ts = pd.to_datetime(predictions.get("actual_observed_at_utc"), errors="coerce", utc=True)
        target_ts = pd.to_datetime(predictions.get("forecast_target_timestamp_utc"), errors="coerce", utc=True)
        same_session = (
            decision_ts.notna()
            & target_ts.notna()
            & (decision_ts.dt.tz_convert(IST).dt.date == target_ts.dt.tz_convert(IST).dt.date)
        )
        predictions = predictions.loc[same_session].copy()
    return predictions, quotes, report


def run_learning(start_date: date, end_date: date, *, output_dir: str | Path = "diagnostics/all_nse_prediction_learning") -> dict[str, Any]:
    store = GoogleDriveStore()
    predictions, quotes, ingest = load_clean_session_data(store, start_date, end_date)
    outcomes = match_matured_outcomes(predictions, quotes)
    if outcomes.empty:
        fit_frame = pd.DataFrame()
    else:
        # Old snapshots predate explicit model-version logging. Keep their scored
        # outcomes in the audit, but never mix them into a version-specific fit.
        fit_frame = outcomes.loc[
            outcomes["validation_status"].eq("VALIDATED")
            & ~outcomes["model_version"].astype(str).isin(["", "LEGACY_UNKNOWN", "None", "nan"])
            & pd.to_numeric(outcomes["probability_up"], errors="coerce").between(0, 1)
            & pd.to_numeric(outcomes["actual_return"], errors="coerce").notna()
        ].copy()
    fit_frame["target_timestamp_utc"] = fit_frame.get("target_timestamp_utc", pd.Series(dtype=object))
    report: dict[str, Any] = {
        "audit_name": "all-nse-production-outcome-learning",
        "schema_version": "all-nse-production-outcome-learning-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "scope": "ALL-NSE pooled production 30-minute classifier/regressor",
        "capture_interval_minutes": 5,
        "partial_data_policy": "Persist partial snapshots for diagnostics, but exclude any slot not passing both full-universe quote and prediction quality gates from outcome fitting.",
        "target_matching_policy": "Nearest clean same-symbol provider quote observation within 180 seconds of nominal +30-minute target; quote observations are not exchange-stamped 1-minute candle closes.",
        "ingestion": ingest,
        "outcome_summary": {
            "prediction_rows_in_clean_slots": int(len(predictions)),
            "matched_validated_outcomes": int(outcomes["validation_status"].eq("VALIDATED").sum()) if not outcomes.empty else 0,
            "pending_or_unmatched_outcomes": int(outcomes["validation_status"].ne("VALIDATED").sum()) if not outcomes.empty else 0,
            "unique_symbols_scored": int(outcomes.loc[outcomes["validation_status"].eq("VALIDATED"), "symbol"].nunique()) if not outcomes.empty else 0,
            "model_versions_in_fit": sorted(fit_frame["model_version"].dropna().astype(str).unique().tolist()) if not fit_frame.empty else [],
            "fit_eligible_rows": int(len(fit_frame)),
            "fit_eligible_sessions": int(pd.to_datetime(fit_frame["target_timestamp_utc"], errors="coerce", utc=True).dt.tz_convert(IST).dt.strftime("%Y-%m-%d").nunique()) if not fit_frame.empty else 0,
        },
        "recommendation_update": {
            "status": "WAITING_FOR_CLEAN_MATURED_OUTCOMES",
            "base_classifier_regressor_weights_modified": False,
            "calibration_profile_auto_applied": False,
            "minimum_autopilot_rows": 600,
            "minimum_autopilot_sessions": 15,
            "activation_policy": "Eligible profiles enter SHADOW, and only profiles that subsequently improve live log loss without worsening Brier score can progress to ACTIVE. Active calibration changes live probability/return/signal outputs; base joblib weights remain unchanged.",
        },
        "production_model_changed": False,
        "orders_placed": False,
    }

    calibration_result = {
        "status": "NOT_RUN_INSUFFICIENT_CLEAN_VERSIONED_ROWS",
        "profiles_live": 0,
        "reason": "No validated outcomes were available for an explicitly versioned full-NSE model, or the versioned clean sample was empty.",
    }
    if not fit_frame.empty:
        # Run through the same persisted registry used by the runtime calibration
        # overlay. Version-specific entries keep this model separate from the
        # watchlist model even though both use the same lifecycle state file.
        runtime_frame = fit_frame.copy()
        runtime_frame["validation_status"] = "VALIDATED"
        runtime_frame["horizon_min"] = 30
        runtime_frame["probability_up"] = pd.to_numeric(runtime_frame["probability_up"], errors="coerce")
        runtime_frame["expected_return"] = pd.to_numeric(runtime_frame["expected_return"], errors="coerce")
        runtime_frame["input_age_seconds"] = pd.to_numeric(runtime_frame["input_age_seconds"], errors="coerce")
        autopilot_dir = Path(output_dir) / "calibration_autopilot"
        calibration_result = _run_calibration_autopilot(store, runtime_frame, autopilot_dir)
        report["recommendation_update"]["status"] = calibration_result.get("status", "UNKNOWN")
        report["recommendation_update"]["calibration_profiles_live"] = calibration_result.get("profiles_live", 0)
        report["recommendation_update"]["base_classifier_regressor_weights_modified"] = False
        report["recommendation_update"]["calibration_profile_auto_applied"] = bool(calibration_result.get("profiles_live", 0))
    report["calibration_autopilot"] = calibration_result

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{start_date:%Y%m%d}_{end_date:%Y%m%d}"
    outcomes_path = out_dir / f"all_nse_outcomes_{stamp}.csv"
    report_path = out_dir / f"all_nse_prediction_learning_report_{stamp}.json"
    outcomes.to_csv(outcomes_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    remote_dir = f"{LEARNING_ROOT}/{end_date:%Y/%m/%d}"
    upload_results = {}
    for key, path, mime_type in [
        ("outcomes", outcomes_path, "text/csv"),
        ("report", report_path, "application/json"),
    ]:
        result = store.upload_or_update(path, f"{remote_dir}/{path.name}", mime_type=mime_type)
        upload_results[key] = result
        if result.get("status") != "OK":
            raise RuntimeError(f"Drive upload failed for {key}: {result}")
    latest = out_dir / "latest_all_nse_learning_report.json"
    latest.write_text(json.dumps({
        "status": report["recommendation_update"]["status"],
        "generated_at_utc": report["generated_at_utc"],
        "date_range": report["date_range"],
        "report_path": f"{remote_dir}/{report_path.name}",
        "outcomes_path": f"{remote_dir}/{outcomes_path.name}",
        "upload_results": upload_results,
        "production_model_changed": False,
        "partial_data_quarantined": True,
    }, indent=2, sort_keys=True), encoding="utf-8")
    latest_result = store.upload_or_update(
        latest, f"{LEARNING_ROOT}/latest_all_nse_learning_report.json",
        mime_type="application/json",
    )
    if latest_result.get("status") != "OK":
        raise RuntimeError(f"Drive latest report upload failed: {latest_result}")
    report["google_drive_sync"] = {"status": "OK", "remote_directory": remote_dir, "uploads": upload_results, "latest_manifest": latest_result}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    final_report_upload = store.upload_or_update(report_path, f"{remote_dir}/{report_path.name}", mime_type="application/json")
    if final_report_upload.get("status") != "OK":
        raise RuntimeError(f"Drive final report update failed: {final_report_upload}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--out-dir", default="diagnostics/all_nse_prediction_learning")
    args = parser.parse_args()
    today = datetime.now(IST).date()
    end_date = date.fromisoformat(args.end_date) if args.end_date else today
    start_date = date.fromisoformat(args.start_date) if args.start_date else end_date - timedelta(days=29)
    if end_date < start_date:
        parser.error("--end-date must be on or after --start-date")
    result = run_learning(start_date, end_date, output_dir=args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
