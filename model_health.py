"""Descriptive model-health monitoring and retraining flags."""
from __future__ import annotations

from typing import Any

import pandas as pd

from probability_calibration import wilson_interval


def assess_delivery_health(validation_df: pd.DataFrame, rolling_window: int = 7, min_samples: int = 7) -> dict[str, Any]:
    if validation_df.empty:
        return {"status": "NO_DATA", "sample_count": 0, "retraining_candidate": False}
    if "validation_status" not in validation_df.columns:
        return {"status": "NO_DATA", "sample_count": 0, "retraining_candidate": False}

    x = validation_df[
        validation_df["validation_status"].astype(str).str.strip().str.upper().eq("VALIDATED")
    ].copy()
    if x.empty:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "sample_count": 0,
            "directional_accuracy": None,
            "mean_absolute_percentage_error": None,
            "retraining_candidate": False,
            "reason": f"Need at least {min_samples} completed delivery predictions for the rolling health gate.",
        }

    if "decision_timestamp" in x.columns:
        x["decision_timestamp"] = pd.to_datetime(x["decision_timestamp"], errors="coerce")
        x = x.dropna(subset=["decision_timestamp"]).sort_values("decision_timestamp").tail(rolling_window)
    else:
        x = x.tail(rolling_window)

    signal = x.get("signal", pd.Series(index=x.index, dtype="object")).astype(str).str.strip().str.upper()
    actual = x.get("actual_direction_d1", pd.Series(index=x.index, dtype="object")).astype(str).str.strip().str.upper()

    matches = pd.Series(
        ((signal == "BUY") & actual.eq("UP"))
        | ((signal == "SELL") & actual.eq("DOWN")),
        index=x.index,
        dtype="float64",
    )
    usable = actual.isin(["UP", "DOWN"]) & signal.isin(["BUY", "SELL"])
    matches = matches[usable].dropna()
    hold_samples = int((signal.eq("HOLD") & actual.isin(["UP", "DOWN", "FLAT"])).sum())

    mape = pd.to_numeric(
        x.get("percentage_error_d1", pd.Series(index=x.index, dtype="float64")),
        errors="coerce",
    ).abs().mean()

    if len(matches) < min_samples:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "sample_count": int(len(matches)),
            "directional_accuracy": float(matches.mean()) if len(matches) else None,
            "directional_accuracy_ci95_low": None,
            "directional_accuracy_ci95_high": None,
            "action_precision": float(matches.mean()) if len(matches) else None,
            "hold_samples": hold_samples,
            "mean_absolute_percentage_error": float(mape) if pd.notna(mape) else None,
            "retraining_candidate": False,
            "retraining_min_samples": max(int(min_samples), 10),
            "reason": f"Need at least {min_samples} completed delivery predictions for the rolling health gate.",
        }

    accuracy = float(matches.mean())
    ci_low, ci_high = wilson_interval(float(matches.sum()), len(matches))

    action = signal.isin(["BUY", "SELL"])
    action_correct = (
        ((signal == "BUY") & actual.eq("UP"))
        | ((signal == "SELL") & actual.eq("DOWN"))
    )
    action_precision = float(action_correct[action].mean()) if bool(action.any()) else None
    action_coverage = float(action.mean()) if len(action) else None

    probability = pd.to_numeric(
        x.get("probability_up_next_session", pd.Series(index=x.index, dtype="float64")),
        errors="coerce",
    )
    actual_up = actual.map({"UP": 1.0, "DOWN": 0.0})
    brier = ((probability - actual_up) ** 2).dropna().mean()

    primary_health = (
        action_precision
        if action_precision is not None and int(action.sum()) >= 10
        else accuracy
    )

    min_retrain_samples = max(int(min_samples), 10)
    return {
        "status": "MONITOR",
        "sample_count": int(len(matches)),
        "directional_accuracy": accuracy,
        "directional_accuracy_ci95_low": ci_low,
        "directional_accuracy_ci95_high": ci_high,
        "action_precision": action_precision,
        "action_coverage": action_coverage,
        "hold_samples": hold_samples,
        "brier_score": float(brier) if pd.notna(brier) else None,
        "mean_absolute_percentage_error": float(mape) if pd.notna(mape) else None,
        "retraining_candidate": bool(len(matches) >= min_retrain_samples and primary_health < 0.50),
        "retraining_min_samples": int(min_retrain_samples),
        "health_metric_used": "action_precision" if action_precision is not None and int(action.sum()) >= 10 else "directional_accuracy",
        "rule": "candidate flag only when the primary OOS action metric falls below 50% with at least %d scored BUY/SELL rows; HOLD rows are reported separately and are never treated as directional misses." % min_retrain_samples,
    }
