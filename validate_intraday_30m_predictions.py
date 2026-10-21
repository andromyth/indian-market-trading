#!/usr/bin/env python3
"""Validate persisted 30-minute model predictions against persisted NSE 1-minute bars.

Only observations at/after the 30-minute target timestamp qualify as actuals.
No model artifacts, prediction signals, or policy thresholds are changed.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from persistent_store import load_persistent_1m
from prediction_ledger import (
    IST_TZ,
    is_within_nse_session,
    load_records,
    load_validation_records,
    save_validation_results,
)


def _root(symbol: Any) -> str:
    value = str(symbol or "").strip().upper()
    for suffix in (".NS", ".BO"):
        if value.endswith(suffix):
            value = value[:-3]
    return value


def _normalise_bars(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["close"])
    x = frame.copy()
    lower_to_original = {str(c).strip().lower(): c for c in x.columns}
    if "timestamp" in lower_to_original:
        idx = pd.to_datetime(x.pop(lower_to_original["timestamp"]), errors="coerce", utc=True)
    elif "datetime" in lower_to_original:
        idx = pd.to_datetime(x.pop(lower_to_original["datetime"]), errors="coerce", utc=True)
    elif "date" in lower_to_original:
        idx = pd.to_datetime(x.pop(lower_to_original["date"]), errors="coerce", utc=True)
    else:
        idx = pd.to_datetime(x.index, errors="coerce", utc=True)
    idx = pd.DatetimeIndex(idx).tz_convert(IST_TZ)
    x.index = idx
    close_col = lower_to_original.get("close")
    if close_col is None:
        close_col = lower_to_original.get("ltp")
    if close_col is None:
        return pd.DataFrame(columns=["close"])
    x["close"] = pd.to_numeric(x[close_col], errors="coerce")
    x = x.loc[~x.index.isna()].dropna(subset=["close"]).sort_index()
    return x[["close"]][~x.index.duplicated(keep="last")]


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(IST_TZ)
    return ts.tz_convert(IST_TZ)


def validate_one(
    record: dict[str, Any],
    bars: pd.DataFrame,
    tolerance_seconds: int = 60,
) -> dict[str, Any]:
    symbol = _root(record.get("symbol"))
    decision = _timestamp(record.get("decision_timestamp"))
    pred_id = str(record.get("prediction_id") or "")
    target = _timestamp(record.get("signal_horizon_end") or record.get("target_timestamp"))
    if decision is None:
        return {
            "prediction_id": pred_id, "symbol": symbol, "mode": "Intraday",
            "prediction_horizon": "30m", "validation_status": "INVALID_PREDICTION",
            "validation_reason": "Missing or invalid decision timestamp.",
        }
    expected_target = decision + pd.Timedelta(minutes=30)
    if target is None:
        target = expected_target
    # Guard against mislabeled/corrupt horizons rather than silently shifting target.
    if abs((target - decision).total_seconds() - 1800) > 1:
        return {
            "prediction_id": pred_id, "symbol": symbol, "mode": "Intraday",
            "prediction_horizon": "30m", "decision_timestamp": decision.isoformat(),
            "target_timestamp": target.isoformat(), "validation_status": "INVALID_PREDICTION",
            "validation_reason": "Stored horizon endpoint is not exactly 30 minutes after decision.",
        }
    base_price = pd.to_numeric(
        record.get("signal_price_reference", record.get("reference_close", record.get("current_price"))),
        errors="coerce",
    )
    p_up = pd.to_numeric(record.get("probability_up_30m", record.get("probability_up")), errors="coerce")
    signal = str(record.get("signal", "HOLD")).strip().upper()
    out: dict[str, Any] = {
        "prediction_id": pred_id,
        "symbol": symbol,
        "mode": "Intraday",
        "prediction_horizon": "30m",
        "model_version": record.get("model_version"),
        "decision_timestamp": decision.isoformat(),
        "target_timestamp": target.isoformat(),
        "target_timestamp_utc": target.tz_convert("UTC").isoformat(),
        "signal": signal,
        "probability_up_30m": float(p_up) if np.isfinite(p_up) else np.nan,
        "validation_status": "WAITING",
    }
    if not pred_id:
        out.update(validation_status="INVALID_PREDICTION", validation_reason="Missing prediction_id.")
        return out
    if not is_within_nse_session(decision, 30):
        out.update(validation_status="OUT_OF_SESSION", validation_reason="30-minute target falls outside the regular NSE cash session.")
        return out
    if bars.empty:
        out["validation_reason"] = "No persisted NSE 1-minute history available."
        return out
    if not np.isfinite(base_price) or float(base_price) <= 0:
        refs = bars.loc[bars.index <= decision, "close"]
        if refs.empty:
            out["validation_reason"] = "No reference close at or before the prediction timestamp."
            return out
        base_price = float(refs.iloc[-1])
    out["signal_price_reference"] = float(base_price)

    # Strictly prohibit a bar before target: first eligible persisted observation
    # at/after target, limited by the alignment tolerance.
    delta_seconds = (bars.index - target).total_seconds()
    eligible = (delta_seconds >= 0) & (delta_seconds <= max(1, int(tolerance_seconds)))
    if not eligible.any():
        out["validation_reason"] = "Target actual bar is not persisted yet or exceeds the alignment tolerance."
        return out

    position = int(np.argmin(np.where(eligible, delta_seconds, np.inf)))
    actual_timestamp = bars.index[position]
    actual_price = float(bars["close"].iloc[position])
    actual_return = actual_price / float(base_price) - 1.0
    actual_up = bool(actual_return > 0)
    predicted_up = bool(float(p_up) >= 0.5) if np.isfinite(p_up) else None
    predicted_price_value = pd.to_numeric(
        record.get("predicted_price_30m", record.get("predicted_price")),
        errors="coerce",
    )
    if not np.isfinite(predicted_price_value):
        expected_return = pd.to_numeric(record.get("expected_return_30m"), errors="coerce")
        if np.isfinite(expected_return):
            predicted_price_value = float(base_price) * (1.0 + float(expected_return))
    signal_match = True if signal == "BUY" and actual_up else (
        True if signal == "SELL" and not actual_up else (
            False if signal in {"BUY", "SELL"} else None
        )
    )
    out.update({
        "actual_timestamp": actual_timestamp.isoformat(),
        "actual_timestamp_utc": actual_timestamp.tz_convert("UTC").isoformat(),
        "target_alignment_seconds": float((actual_timestamp - target).total_seconds()),
        "realized_horizon_seconds": float((actual_timestamp - decision).total_seconds()),
        "actual_price_30m": actual_price,
        "actual_return_30m": float(actual_return),
        "actual_direction_30m": "UP" if actual_up else "DOWN",
        "predicted_direction_30m": "UP" if predicted_up is True else ("DOWN" if predicted_up is False else None),
        "direction_match_30m": (predicted_up == actual_up) if predicted_up is not None else None,
        "signal_direction_match_30m": signal_match,
        "brier_score_30m": (float(p_up) - float(actual_up)) ** 2 if np.isfinite(p_up) else np.nan,
        "predicted_price_30m": float(predicted_price_value) if np.isfinite(predicted_price_value) else np.nan,
        "abs_price_error_30m": abs(float(predicted_price_value) - actual_price) if np.isfinite(predicted_price_value) else np.nan,
        "validation_status": "VALIDATED",
    })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance-seconds", type=int, default=60)
    parser.add_argument("--no-sync-drive", action="store_true")
    args = parser.parse_args()

    all_records = load_records()
    predictions = [
        row for row in all_records
        if str(row.get("mode", "")).strip().lower() == "intraday"
        and str(row.get("prediction_horizon", "")).strip().lower().replace(" ", "") in {"30m", "30min", "30minutes"}
        and row.get("prediction_id")
    ]
    if not predictions:
        print(json.dumps({"status": "NO_30M_PREDICTIONS", "predictions_seen": 0}, indent=2))
        return 0

    existing: dict[str, dict[str, Any]] = {}
    for row in load_validation_records():
        if row.get("prediction_id"):
            existing[str(row["prediction_id"])] = row
    completed_ids = {
        pid for pid, row in existing.items()
        if str(row.get("validation_status", "")).upper() == "VALIDATED"
        and pd.notna(pd.to_numeric(row.get("actual_return_30m"), errors="coerce"))
        and pd.notna(pd.to_numeric(row.get("actual_direction_30m") == "UP", errors="coerce"))
    }
    pending = [row for row in predictions if str(row["prediction_id"]) not in completed_ids]
    if not pending:
        print(json.dumps({
            "status": "NO_UNRESOLVED_30M_PREDICTIONS",
            "predictions_seen": len(predictions),
            "already_validated": len(completed_ids),
            "saved": False,
        }, indent=2))
        return 0

    bar_cache: dict[str, pd.DataFrame] = {}
    results = []
    for record in pending:
        symbol = _root(record.get("symbol"))
        if symbol not in bar_cache:
            bar_cache[symbol] = _normalise_bars(load_persistent_1m(symbol))
        results.append(validate_one(record, bar_cache[symbol], args.tolerance_seconds))

    counts = pd.Series([row.get("validation_status", "UNKNOWN") for row in results]).value_counts().to_dict()
    validated = [row for row in results if row.get("validation_status") == "VALIDATED"]
    saved = save_validation_results(results, sync_drive=not args.no_sync_drive)
    action_matches = [
        bool(row["signal_direction_match_30m"])
        for row in validated if row.get("signal") in {"BUY", "SELL"} and row.get("signal_direction_match_30m") is not None
    ]
    report = {
        "status": "PASS" if (args.no_sync_drive or saved.get("synced") is True) else "PERSISTENCE_ERROR",
        "predictions_seen": len(predictions),
        "pending_scored": len(pending),
        "validation_status_counts": counts,
        "validated": len(validated),
        "action_direction_accuracy_descriptive": float(np.mean(action_matches)) if action_matches else None,
        "action_predictions_scored": len(action_matches),
        "saved_validation_rows": saved.get("rows"),
        "drive_synced": saved.get("synced"),
        "warning": "Minute-level targets overlap. These are descriptive out-of-sample outcomes, not a portfolio backtest or net-of-cost results.",
    }
    print(json.dumps(report, indent=2, default=str))
    return 0 if (args.no_sync_drive or saved.get("synced") is True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
