"""Causal prediction-vs-actual validation for the delivery prediction ledger.

A next-session delivery model has one target-aligned outcome: D+1. To satisfy
continuous seven-day monitoring without misrepresenting the model target, the
validator reports the rolling performance of the last seven completed daily
predictions. It also records D+1 through D+7 path observations as monitoring,
not as additional model-target accuracy claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationConfig:
    rolling_days: int = 7
    hold_tolerance_pct: float = 0.005
    minimum_health_samples: int = 7


def _norm_history(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if not isinstance(x.index, pd.DatetimeIndex):
        time_col = next((c for c in ["Date", "Datetime", "timestamp", "date"] if c in x.columns), None)
        if time_col is None:
            raise ValueError("History requires a DatetimeIndex or Date/Datetime/timestamp/date column")
        idx = pd.to_datetime(x[time_col], errors="coerce")
        x = x.drop(columns=[time_col])
    else:
        idx = pd.to_datetime(x.index, errors="coerce")
    x.index = idx
    x = x[~x.index.isna()].sort_index()
    return x


def _direction(actual_return: float, tol: float) -> str:
    if not np.isfinite(actual_return):
        return "UNKNOWN"
    t = tol / 100.0
    if actual_return > t:
        return "UP"
    if actual_return < -t:
        return "DOWN"
    return "FLAT"


def validate_delivery_prediction(record: dict[str, Any], history_df: pd.DataFrame, config: ValidationConfig | None = None) -> dict[str, Any]:
    cfg = config or ValidationConfig()
    x = _norm_history(history_df)
    ts = pd.Timestamp(record.get("decision_timestamp"))
    if ts.tzinfo is not None and getattr(x.index, "tz", None) is None:
        x.index = x.index.tz_localize(ts.tz)
    elif ts.tzinfo is None and getattr(x.index, "tz", None) is not None:
        ts = ts.tz_localize(x.index.tz)
    close = pd.to_numeric(x.get("Close"), errors="coerce")
    ref_rows = x.loc[x.index <= ts]
    if ref_rows.empty:
        return {"prediction_id": record.get("prediction_id"), "validation_status": "WAITING"}
    ref_close = float(close.loc[ref_rows.index[-1]])
    future = x.loc[x.index > ref_rows.index[-1]].copy()
    future["Close"] = pd.to_numeric(future["Close"], errors="coerce")
    future = future.dropna(subset=["Close"]).head(max(cfg.rolling_days, 7))
    if future.empty:
        return {"prediction_id": record.get("prediction_id"), "validation_status": "WAITING"}

    predicted_return = float(record.get("predicted_return_next_session", np.nan))
    predicted_price = ref_close * (1.0 + predicted_return) if np.isfinite(predicted_return) else np.nan
    actual_d1_close = float(future["Close"].iloc[0])
    actual_d1_return = actual_d1_close / ref_close - 1.0
    signal = str(record.get("signal", "HOLD")).upper()
    actual_direction = _direction(actual_d1_return, cfg.hold_tolerance_pct)
    direction_match = None
    if signal == "BUY": direction_match = actual_direction == "UP"
    elif signal == "SELL": direction_match = actual_direction == "DOWN"
    elif signal == "HOLD": direction_match = actual_direction == "FLAT"

    out: dict[str, Any] = {
        "prediction_id": record.get("prediction_id"),
        "symbol": record.get("symbol"),
        "decision_timestamp": record.get("decision_timestamp"),
        "signal": signal,
        "predicted_return_next_session": predicted_return,
        "predicted_price_next_session": predicted_price,
        "actual_d1_close": actual_d1_close,
        "actual_d1_return": actual_d1_return,
        "absolute_price_error_d1": actual_d1_close - predicted_price if np.isfinite(predicted_price) else np.nan,
        "percentage_error_d1": (actual_d1_close / predicted_price - 1.0) if np.isfinite(predicted_price) and predicted_price else np.nan,
        "actual_direction_d1": actual_direction,
        "direction_match_d1": direction_match,
        "validation_status": "VALIDATED",
    }
    # Monitoring path. These fields deliberately describe follow-through after
    # the target-aligned D+1 observation, not additional target accuracy.
    for i, (idx, row) in enumerate(future.head(7).iterrows(), start=1):
        c = float(row["Close"])
        out[f"d{i}_date"] = str(idx)
        out[f"d{i}_actual_close"] = c
        out[f"d{i}_return_from_decision"] = c / ref_close - 1.0
        out[f"d{i}_distance_from_predicted_d1_price"] = c - predicted_price if np.isfinite(predicted_price) else np.nan
    return out


def validate_ledger(records: list[dict[str, Any]], history_provider: Callable[[str], pd.DataFrame], config: ValidationConfig | None = None) -> pd.DataFrame:
    cfg = config or ValidationConfig()
    rows = []
    for r in records:
        if str(r.get("mode", "")).lower() != "delivery":
            continue
        try:
            rows.append(validate_delivery_prediction(r, history_provider(str(r["symbol"])), cfg))
        except Exception as exc:
            rows.append({"prediction_id": r.get("prediction_id"), "symbol": r.get("symbol"), "validation_status": "ERROR", "validation_error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def rolling_7day_scorecard(validation_df: pd.DataFrame) -> dict[str, Any]:
    if validation_df.empty:
        return {"status": "NO_DATA"}
    x = validation_df[validation_df["validation_status"] == "VALIDATED"].copy()
    x = x.sort_values("decision_timestamp").tail(7)
    matches = pd.to_numeric(x.get("direction_match_d1"), errors="coerce")
    valid_matches = matches.dropna()
    mae = pd.to_numeric(x.get("percentage_error_d1"), errors="coerce").abs().mean()
    return {
        "status": "PASS" if len(x) else "NO_DATA",
        "completed_predictions": int(len(x)),
        "directional_accuracy_7_predictions": float(valid_matches.mean()) if len(valid_matches) else None,
        "mean_absolute_percentage_price_error": float(mae) if pd.notna(mae) else None,
        "latest_decision_date": str(x["decision_timestamp"].iloc[-1]) if len(x) else None,
    }
