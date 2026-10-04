#!/usr/bin/env python3
"""Validate persisted 5-minute-step Intraday predictions against actual NSE 1-minute closes.

This validator is read-only with respect to model artifacts. It records validation
results in the existing prediction-validation ledger and never retrains or
modifies a prediction model.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from persistent_store import load_persistent_1m
from prediction_ledger import load_records, save_validation_results


def _root(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    for suffix in (".NS", ".BO"):
        if s.endswith(suffix):
            s = s[:-3]
    return s


def _norm(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    x = frame.copy()
    x.columns = [str(c).strip() for c in x.columns]
    aliases = {
        "timestamp": "timestamp", "datetime": "timestamp",
        "date": "timestamp", "time": "timestamp", "ltp": "close",
        "close": "close",
    }
    x = x.rename(columns={c: aliases.get(str(c).lower(), c) for c in x.columns})
    if "timestamp" in x.columns:
        idx = pd.to_datetime(x.pop("timestamp"), errors="coerce", utc=True)
    else:
        idx = pd.to_datetime(x.index, errors="coerce", utc=True)
    idx = pd.DatetimeIndex(idx).tz_convert("Asia/Kolkata")
    x.index = idx
    if "close" not in x.columns:
        return pd.DataFrame()
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x = x.dropna(subset=["close"]).sort_index()
    return x[~x.index.duplicated(keep="last")]


def _horizon(record: dict) -> int:
    value = record.get("horizon_minutes")
    if value is not None:
        return int(value)
    raw = str(record.get("prediction_horizon", "30m"))
    match = re.search(r"(\d+)", raw)
    return int(match.group(1)) if match else 30


def validate_one(record: dict, tolerance_seconds: int = 60) -> dict:
    symbol = _root(record.get("symbol"))
    ts = pd.Timestamp(record.get("decision_timestamp"))
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Kolkata")
    else:
        ts = ts.tz_convert("Asia/Kolkata")

    horizon = _horizon(record)
    target_ts = ts + pd.Timedelta(minutes=horizon)
    frame = _norm(load_persistent_1m(symbol))
    base_price = pd.to_numeric(
        record.get("current_price", record.get("prediction_price_reference")),
        errors="coerce",
    )

    out = {
        "prediction_id": record.get("prediction_id"),
        "symbol": symbol,
        "mode": "Intraday",
        "prediction_horizon": f"{horizon}m",
        "decision_timestamp": str(ts),
        "target_timestamp": str(target_ts),
        "signal": str(record.get("signal", "HOLD")).upper(),
        "predicted_price": pd.to_numeric(record.get("predicted_price"), errors="coerce"),
        "probability_up": pd.to_numeric(
            record.get("probability_up", record.get("probability_up_30m")),
            errors="coerce",
        ),
        "validation_status": "WAITING",
    }

    if frame.empty:
        out["validation_reason"] = "No persisted NSE 1-minute history available."
        return out

    if not np.isfinite(base_price):
        ref = frame.loc[frame.index <= ts, "close"]
        if ref.empty:
            out["validation_reason"] = "No reference close at or before the decision timestamp."
            return out
        base_price = float(ref.iloc[-1])

    # Actual outcome must be at or after the nominal target. Never validate
    # a +5m prediction using a pre-target bar merely because it is close in time.
    delta_seconds=(frame.index-target_ts).total_seconds()
    eligible=(
        (delta_seconds>=0)
        & (delta_seconds<=max(1,int(tolerance_seconds)))
    )
    if not eligible.any():
        out["validation_reason"] = "Target actual bar is not persisted yet."
        return out

    nearest_position = int(np.argmin(np.where(eligible, delta, np.inf)))
    actual_ts = frame.index[nearest_position]
    actual_price = float(frame["close"].iloc[nearest_position])
    alignment_seconds=float((actual_ts-target_ts).total_seconds())
    actual_return = actual_price / float(base_price) - 1.0 if base_price else np.nan
    actual_direction = "UP" if actual_return > 0 else "DOWN"

    signal = out["signal"]
    direction_match = None
    if signal == "BUY":
        direction_match = actual_return > 0
    elif signal == "SELL":
        direction_match = actual_return < 0

    predicted_price = out["predicted_price"]
    price_error = (
        float(predicted_price - actual_price)
        if np.isfinite(predicted_price) else np.nan
    )
    p_up = out["probability_up"]
    actual_label = 1.0 if actual_return > 0 else 0.0
    brier = (float(p_up) - actual_label) ** 2 if np.isfinite(p_up) else np.nan

    out.update({
        "prediction_price_reference": float(base_price),
        "actual_timestamp": str(actual_ts),
        "target_alignment_seconds": alignment_seconds,
        "realized_horizon_seconds": float((actual_ts-ts).total_seconds()),
        "actual_price": actual_price,
        "actual_return": actual_return,
        "actual_direction": actual_direction,
        "direction_match": direction_match,
        "price_error": price_error,
        "abs_price_error": abs(price_error) if np.isfinite(price_error) else np.nan,
        "brier_score": brier,
        "validation_status": "VALIDATED",
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--tolerance-seconds", type=int, default=60)
    ap.add_argument("--no-sync-drive", action="store_true")
    args = ap.parse_args()

    records = [
        r for r in load_records()
        if str(r.get("model_version", "")) in {"2026-5m-step-multihorizon","2026-5m-step-multihorizon-v2"}
        and str(r.get("mode", "")).lower() == "intraday"
    ]
    if args.symbols.strip():
        allowed = {_root(x) for x in args.symbols.split(",") if x.strip()}
        records = [r for r in records if _root(r.get("symbol")) in allowed]

    if not records:
        print("No 5-minute-step Intraday predictions found.")
        return 0

    results = [validate_one(r, args.tolerance_seconds) for r in records]
    validation = pd.DataFrame(results)
    saved = save_validation_results(
        validation.to_dict("records"),
        sync_drive=not args.no_sync_drive,
    )
    complete = validation[validation["validation_status"].eq("VALIDATED")].copy()

    print("5-MINUTE INTRADAY VALIDATION")
    print("predictions_seen:", len(records))
    print("validated:", len(complete))
    print("waiting:", int((validation["validation_status"] == "WAITING").sum()))
    if not complete.empty:
        action = complete["signal"].isin(["BUY", "SELL"])
        action_accuracy = (
            pd.to_numeric(complete.loc[action, "direction_match"], errors="coerce").mean()
            if action.any() else np.nan
        )
        print("action_direction_accuracy:", float(action_accuracy) if np.isfinite(action_accuracy) else None)
        print("mean_abs_price_error:", float(complete["abs_price_error"].mean()))
        print("mean_brier_score:", float(complete["brier_score"].mean()))
    print("save_result:", saved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
