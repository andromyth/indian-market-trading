#!/usr/bin/env python3
"""Generate one idempotent ALL-NSE Delivery prediction batch from the PIT snapshot.

This stage scores the already-built 59-feature production snapshot. It does not
retrain, promote, or modify model artifacts. Repeated identical outputs are
idempotent; changed output for the same symbol/date/model is stored as a
separate, auditable prediction revision.
"""
from __future__ import annotations

import argparse
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from all_nse_delivery_model import _load_models, _signal_policy, _safe_float
from delivery_prediction_csv_ledger import (
    load_eod_history,
    project_next_session_ohlc,
    sync_delivery_csv_ledger,
)
from prediction_ledger import append_predictions
from delivery_quality_gate import apply_quality_gate, quality_gate_reasons


DEFAULT_BASE = "/content/drive/MyDrive/Indian_Market_Research"
BASE = Path(os.getenv("INDIAN_MARKET_RESEARCH_ROOT", DEFAULT_BASE))
FEATURE_DIR = BASE / "features" / "daily"


def _latest_snapshot_path(snapshot_date: str | None) -> tuple[Path, str]:
    if snapshot_date:
        path = FEATURE_DIR / f"DELIVERY_PIT_FEATURES_{snapshot_date}.parquet"
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"PIT snapshot not found: {path}")
        return path, snapshot_date

    candidates: list[tuple[date, Path]] = []
    rx = re.compile(r"DELIVERY_PIT_FEATURES_(20\d{2}-\d{2}-\d{2})\.parquet$")
    for path in FEATURE_DIR.glob("DELIVERY_PIT_FEATURES_*.parquet"):
        m = rx.match(path.name)
        if not m:
            continue
        try:
            d = pd.Timestamp(m.group(1)).date()
        except Exception:
            continue
        candidates.append((d, path))

    if not candidates:
        raise FileNotFoundError(f"No DELIVERY_PIT_FEATURES_*.parquet snapshots found in {FEATURE_DIR}")

    d, path = max(candidates, key=lambda x: x[0])
    return path, d.isoformat()


def _next_trading_day(value: date) -> date:
    # Keep the target-date calculation aligned with the existing 2026 NSE
    # holiday calendar used by the ingestion/freshness layer.
    from production_data_freshness_gate import HOLIDAYS
    candidate = value + timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in HOLIDAYS:
        candidate += timedelta(days=1)
    return candidate


def _previous_trading_day(value: date) -> date:
    from production_data_freshness_gate import HOLIDAYS
    candidate = value - timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in HOLIDAYS:
        candidate -= timedelta(days=1)
    return candidate


def _as_bool_gate(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "ready"}
    try:
        return bool(int(value))
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-date", default=None)
    ap.add_argument(
        "--prediction-date",
        default=None,
        help="NSE session the prediction is for (YYYY-MM-DD). "
             "When supplied, the prior trading session snapshot is used.",
    )
    ap.add_argument("--no-sync-drive", action="store_true")
    args = ap.parse_args()

    if args.prediction_date:
        target_date = date.fromisoformat(args.prediction_date)
        inferred_snapshot = _previous_trading_day(target_date)
        snapshot_path, snapshot_date = _latest_snapshot_path(
            args.snapshot_date or inferred_snapshot.isoformat()
        )
        prediction_for_date = target_date.isoformat()
    else:
        snapshot_path, snapshot_date = _latest_snapshot_path(args.snapshot_date)
        prediction_for_date = _next_trading_day(
            date.fromisoformat(snapshot_date)
        ).isoformat()
    pit = pd.read_parquet(snapshot_path)
    if pit.empty:
        raise RuntimeError(f"PIT snapshot is empty: {snapshot_path}")
    if "symbol" not in pit.columns:
        raise RuntimeError("PIT snapshot lacks symbol column.")

    clf, reg, manifest = _load_models()
    features = list(manifest.get("features", []))
    if len(features) != 59:
        raise RuntimeError(f"Locked Delivery model exposes {len(features)} features; expected 59.")

    missing = [c for c in features if c not in pit.columns]
    if missing:
        raise RuntimeError(f"PIT snapshot is missing locked model features: {missing}")

    x = pit.copy()
    x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
    x = x[x["symbol"].ne("")].drop_duplicates("symbol", keep="last").reset_index(drop=True)
    if x.empty:
        raise RuntimeError("No symbols remain after normalization.")

    matrix = x[features].apply(pd.to_numeric, errors="coerce")
    p_up = np.asarray(clf.predict_proba(matrix)[:, 1], dtype=float)
    expected_return = np.asarray(reg.predict(matrix), dtype=float)

    policy = _signal_policy()
    buy_p = float(policy["buy_probability_threshold"])
    sell_p = float(policy["sell_probability_threshold"])
    min_expected_return = float(policy["min_expected_return"])

    raw_gate = x["production_data_quality_gate"] if "production_data_quality_gate" in x.columns else pd.Series(False, index=x.index)
    gate = raw_gate.map(_as_bool_gate).to_numpy(dtype=bool)

    raw_signals = np.where(
        (p_up >= buy_p) & (expected_return > min_expected_return),
        "BUY",
        np.where(
            (p_up <= sell_p) & (expected_return < -min_expected_return),
            "SELL",
            "HOLD",
        ),
    )
    gate_reasons_by_row = []
    for _, row in x.iterrows():
        reasons = quality_gate_reasons(row)
        if not gate[len(gate_reasons_by_row)] and not reasons:
            reasons = ["QUALITY_GATE_BLOCKED_UNSPECIFIED"]
        gate_reasons_by_row.append(reasons)
    # A blocked BUY/SELL is not a HOLD: HOLD means the model was allowed to
    # score the row and found no action. Keep the raw model output for audit,
    # but publish DATA INSUFFICIENT when the strict quality gate fails.
    signals = np.asarray(apply_quality_gate(raw_signals, gate), dtype=object)

    decision_ts = f"{snapshot_date}T15:30:00+05:30"
    model_version = str(manifest.get("engine_version", "2026-all-nse-delivery-multifactor-59-replacement-v1"))
    universe_count = int(manifest.get("universe_symbol_count", len(x)))
    trainable_count = int(manifest.get("trainable_symbol_count", len(x)))
    excluded_count = int(manifest.get("excluded_symbol_count", max(0, universe_count - trainable_count)))

    # Diagnostic only: quantify the fitted model's input-group influence on this
    # prediction batch. It does not modify model outputs or the BUY/SELL/HOLD policy.
    factor_weightage_report = None
    try:
        from delivery_factor_weightage import build_factor_weightage_report
        factor_weightage_report = build_factor_weightage_report(
            classifier=clf,
            regressor=reg,
            matrix=matrix,
            symbols=x["symbol"].astype(str).tolist(),
            probability_up=p_up,
            predicted_return=expected_return,
            raw_signals=raw_signals,
            final_signals=signals,
            quality_gate=gate,
            snapshot_date=snapshot_date,
            prediction_date=prediction_for_date,
            model_version=model_version,
            signal_policy=policy,
        )
        classifier_summary = factor_weightage_report.get("classifier", {})
        regressor_summary = factor_weightage_report.get("regressor", {})
        print("FACTOR_WEIGHTAGE_AUDIT_SUMMARY:", __import__("json").dumps({
            "status": factor_weightage_report.get("status"),
            "snapshot_date": snapshot_date,
            "prediction_for_date": prediction_for_date,
            "classifier_group_mean_abs_logit_share_pct": {
                name: detail.get("mean_per_symbol_abs_logit_share_pct")
                for name, detail in classifier_summary.get("parameter_weightage", {}).items()
            },
            "classifier_coefficient_magnitude_share_pct": {
                name: detail.get("coefficient_magnitude_share_pct")
                for name, detail in classifier_summary.get("parameter_weightage", {}).items()
            },
            "regressor_tree_split_gain_share_pct": regressor_summary.get("tree_split_gain_share_pct"),
            "regressor_group_counterfactual_share_pct": {
                name: detail.get("mean_per_symbol_abs_counterfactual_share_pct")
                for name, detail in regressor_summary.get("current_snapshot_group_sensitivity", {}).items()
            },
            "quality_gate": factor_weightage_report.get("data_quality_audit"),
            "news_semantics": factor_weightage_report.get("news_semantics"),
        }, default=str), flush=True)
    except Exception as exc:
        # Never suppress or modify a prediction because a separate introspection
        # diagnostic failed. The normal persisted-prediction checks remain authoritative.
        print(f"FACTOR_WEIGHTAGE_AUDIT_ERROR (non-blocking): {type(exc).__name__}: {exc}", flush=True)

    # Read the daily EOD archive only once. Re-reading and concatenating all
    # Parquet files for every symbol made this step scale as symbols x files.
    ohlc_history = load_eod_history()
    if not ohlc_history.empty and "symbol" in ohlc_history.columns:
        ohlc_history_by_symbol = {
            str(symbol).strip().upper(): history
            for symbol, history in ohlc_history.groupby("symbol", sort=False)
        }
    else:
        ohlc_history_by_symbol = {}
    empty_ohlc_history = ohlc_history.iloc[0:0]

    records: list[dict[str, Any]] = []
    for i, row in x.iterrows():
        symbol = str(row["symbol"])
        non_missing = int(matrix.iloc[i].notna().sum())
        records.append(
            {
                "symbol": symbol,
                "mode": "Delivery",
                "prediction_horizon": "next_session",
                "prediction_for_date": prediction_for_date,
                "model_version": model_version,
                "engine_version": model_version,
                "decision_engine_version": model_version,
                "decision_timestamp": decision_ts,
                "reference_session": snapshot_date,
                "pit_snapshot_date": snapshot_date,
                "prediction_status": "PASS",
                "probability_up_next_session": _safe_float(p_up[i]),
                "base_model_probability_up_next_session": _safe_float(p_up[i]),
                "probability_down_next_session": _safe_float(1.0 - p_up[i]),
                "predicted_return_next_session": _safe_float(expected_return[i]),
                "reference_open": _safe_float(row.get("open", np.nan)),
                "reference_high": _safe_float(row.get("high", np.nan)),
                "reference_low": _safe_float(row.get("low", np.nan)),
                "reference_close": _safe_float(row.get("close", np.nan)),
                "predicted_price_next_session": (
                    _safe_float(row.get("close", np.nan)) * (1.0 + _safe_float(expected_return[i]))
                    if np.isfinite(_safe_float(row.get("close", np.nan)))
                    and np.isfinite(_safe_float(expected_return[i]))
                    else np.nan
                ),
                "predicted_price_change_next_session": (
                    _safe_float(row.get("close", np.nan)) * _safe_float(expected_return[i])
                    if np.isfinite(_safe_float(row.get("close", np.nan)))
                    and np.isfinite(_safe_float(expected_return[i]))
                    else np.nan
                ),

                "signal": str(signals[i]),
                "raw_model_signal": str(raw_signals[i]),
                "signal_status": "DATA_INSUFFICIENT" if not gate[i] else "SCORED",
                "signal_reason": ";".join(gate_reasons_by_row[i]),
                "quality_gate_block_reasons": gate_reasons_by_row[i],
                "confidence": (
                    "HIGH" if abs(p_up[i] - 0.5) >= 0.25
                    else "MEDIUM" if abs(p_up[i] - 0.5) >= 0.12
                    else "LOW"
                ),
                "feature_count": 59,
                "non_missing_feature_count": non_missing,
                "feature_non_missing_fraction": non_missing / 59.0,
                "data_readiness_score": non_missing / 59.0,
                "feature_source": "PERSISTED_PIT_SNAPSHOT",
                "production_data_quality_gate": bool(gate[i]),
                "production_data_quality_score": _safe_float(row.get("production_data_quality_score", np.nan)),
                "signal_policy": policy,
                "universe_symbol_count": universe_count,
                "trainable_symbol_count": trainable_count,
                "excluded_symbol_count": excluded_count,
                "minimum_history_sessions": int(manifest.get("minimum_history_sessions", 60)),
            }
        )
        # The statistical OHLC projection is persisted with the prediction so
        # the daily CSV ledger and Streamlit display use the same forecast.
        records[-1]["predicted_open"] = np.nan
        records[-1]["predicted_high"] = np.nan
        records[-1]["predicted_low"] = np.nan
        records[-1]["ohlc_prediction_method"] = ""
        symbol_history = ohlc_history_by_symbol.get(
            symbol.strip().upper(), empty_ohlc_history
        )
        ohlc = project_next_session_ohlc(
            symbol,
            snapshot_date,
            row.get("close", np.nan),
            records[-1]["predicted_price_next_session"],
            eod_history=symbol_history,
        )
        if ohlc.get("status") == "PASS":
            records[-1]["predicted_open"] = _safe_float(ohlc.get("predicted_open"))
            records[-1]["predicted_high"] = _safe_float(ohlc.get("predicted_high"))
            records[-1]["predicted_low"] = _safe_float(ohlc.get("predicted_low"))
            records[-1]["ohlc_prediction_method"] = str(ohlc.get("ohlc_prediction_method", ""))


    sync_drive = not args.no_sync_drive
    result = append_predictions(records, sync_drive=sync_drive)
    # A successful process exit must mean the JSONL prediction ledger reached
    # Drive. A failed upload is a hard failure so the workflow retries.
    if sync_drive and not result.get("synced"):
        raise RuntimeError(
            "The local ALL-NSE Delivery prediction ledger could not be synced "
            f"to Google Drive (including any earlier pending rows): {result}"
        )

    # Rebuild the one-row-per-symbol/date CSV view from the authoritative
    # revision-aware JSONL ledger on every synced run. This also repairs a stale
    # CSV after an interrupted earlier upload, including when no new rows append.
    csv_sync_result = None
    if sync_drive:
        csv_sync_result = sync_delivery_csv_ledger(eod_history=ohlc_history)
        print(
            "DELIVERY_PREDICTION_CSV_LEDGER_SYNC:",
            __import__("json").dumps(csv_sync_result, default=str),
            flush=True,
        )
        if csv_sync_result.get("status") != "SYNCED":
            raise RuntimeError(
                "Delivery prediction JSONL ledger is available, but the "
                f"latest-revision CSV view did not sync to Drive: {csv_sync_result}"
            )

    if factor_weightage_report is not None:
        try:
            import json
            from persistent_store import GoogleDriveStore
            report_path = FEATURE_DIR / f"DELIVERY_FACTOR_WEIGHTAGE_AUDIT_{snapshot_date}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(factor_weightage_report, indent=2, default=str),
                encoding="utf-8",
            )
            remote_report_path = f"features/daily/{report_path.name}"
            if sync_drive:
                upload_result = GoogleDriveStore().upload_or_update(
                    report_path, remote_report_path, mime_type="application/json"
                )
                drive_action = upload_result.get("action")
            else:
                drive_action = "SKIPPED_NO_SYNC_DRIVE"
            print("FACTOR_WEIGHTAGE_AUDIT_PERSISTED:", json.dumps({
                "local_path": str(report_path),
                "remote_path": remote_report_path if sync_drive else None,
                "symbols": len(factor_weightage_report.get("per_symbol_predictions", [])),
                "drive_action": drive_action,
            }, default=str), flush=True)
        except Exception as exc:
            # Diagnostics are deliberately non-blocking; signal and ledger
            # persistence were already handled independently above.
            print(f"FACTOR_WEIGHTAGE_AUDIT_UPLOAD_ERROR (non-blocking): {type(exc).__name__}: {exc}", flush=True)

    signal_counts = pd.Series(signals).value_counts().to_dict()
    print("ALL-NSE DELIVERY PREDICTION CYCLE")
    print("snapshot_date:", snapshot_date)
    print("prediction_for_date:", prediction_for_date)
    print("snapshot_path:", snapshot_path)
    print("symbols_scored:", len(records))
    print("production_quality_gate_true:", int(gate.sum()))
    print("signals:", signal_counts)
    print("append_result:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
